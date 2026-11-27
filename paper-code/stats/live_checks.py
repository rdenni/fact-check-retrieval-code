"""
live_checks.py -- ask the public CimpleKG endpoint, today, whether the defects
we found in our snapshot are still there, and save the answers with the date.

The snapshot is fixed, so every number about it is reproducible forever from the
mirror file. The live graph changes every day, so a sentence like "on the public
endpoint, on 12 November 2026, this still held" can only be reproduced as "the
script asked on that date and saved the answer". This script asks and saves; the
paper quotes the saved answer and its date (latex/figures/live_numbers.tex).
Running it again later may give other answers: that is the point of the date.

The checks (each one SPARQL query, read-only, one at a time, with a pause):
  1. entity linking of the explanations, per month, English fact-checks, --english-from to --english-to:
     how many fact-checks, how many with at least one schema:mentions on the ClaimReview node;
  2. the same for the fact-checks of every language, the last --recent-months months up to today;
  3. the claim-level entities of the English fact-checks: how many distinct entities, the most frequent one;
  4. the last month of publication of some publisher names (--publishers).

Outputs:
  --out/live_numbers.tex      \\newcommand macros (default latex/figures)
  --report/live_checks.json   every query and its raw answer, with the date and time (default WORK/stats)
  --report/live_checks.md     the same, for reading

Usage: python stats/live_checks.py [--work WORK] [--endpoint https://data.cimple.eu/sparql] [--out DIR] [--report DIR]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonl import write_json  # noqa: E402
from stats.plot_archive import MONTHS, tex_macro  # noqa: E402

PREFIX = "PREFIX sc: <http://schema.org/>\n"
PAUSE_SECONDS = 2
USER_AGENT = "research-snapshot-check/1.0 (read-only SPARQL counts)"


def month_words(m: str) -> str:
    return f"{MONTHS[int(m[5:7]) - 1]} {m[:4]}" if m else "none"


def ask(endpoint: str, query: str, timeout: int = 300) -> list[dict]:
    """Run one SELECT query and return its rows as dicts of strings (CSV results)."""
    for attempt in range(3):
        try:
            r = requests.post(endpoint, data={"query": query}, timeout=timeout,
                              headers={"Accept": "text/csv", "User-Agent": USER_AGENT})
            r.raise_for_status()
            return list(csv.DictReader(io.StringIO(r.text)))
        except requests.RequestException as e:
            if attempt == 2:
                raise SystemExit(f"the endpoint did not answer: {e}")
            time.sleep(10 * (attempt + 1))
    return []


def per_month_query(month_from: str, month_to: str, english_only: bool) -> str:
    lang = '?rev sc:inLanguage ?lang . FILTER (STR(?lang) = "English")\n' if english_only else ""
    return PREFIX + (
        "SELECT ?m (COUNT(DISTINCT ?rev) AS ?n) (COUNT(DISTINCT ?withEnt) AS ?withEntities) WHERE {\n"
        "  ?rev a sc:ClaimReview ; sc:datePublished ?d .\n  " + lang +
        "  BIND (SUBSTR(STR(?d), 1, 7) AS ?m)\n"
        f'  FILTER (?m >= "{month_from}" && ?m <= "{month_to}")\n'
        "  OPTIONAL { ?rev sc:mentions ?e . BIND (?rev AS ?withEnt) }\n"
        "} GROUP BY ?m ORDER BY ?m")


def shift_month(m: str, k: int) -> str:
    y, mo = int(m[:4]), int(m[5:7]) - 1 + k
    return f"{y + mo // 12:04d}-{mo % 12 + 1:02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="https://data.cimple.eu/sparql")
    ap.add_argument("--work", default=None, help="the report goes to WORK/stats unless --report is given")
    ap.add_argument("--report", default=None)
    ap.add_argument("--out", default=None, help="default: <repository>/latex/figures")
    ap.add_argument("--english-from", default="2024-12")
    ap.add_argument("--english-to", default="2025-06")
    ap.add_argument("--recent-months", type=int, default=6)
    ap.add_argument("--publishers", default="politifact,snopescom,snopes.com,factcheckorg,the-washington-post-fact-checker")
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent.parent / "latex" / "figures"
    report = Path(args.report) if args.report else (Path(args.work) / "stats" if args.work else out)
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    this_month = now.strftime("%Y-%m")
    checks = []

    def run(name, query):
        rows = ask(args.endpoint, query)
        checks.append({"name": name, "query": query, "rows": rows})
        print(f"{name}: {len(rows)} rows")
        time.sleep(PAUSE_SECONDS)
        return rows

    eng = run("English fact-checks per month, with explanation-level entities",
              per_month_query(args.english_from, args.english_to, True))
    recent_from = shift_month(this_month, -(args.recent_months - 1))
    allm = run("all fact-checks per month (every language), with explanation-level entities",
               per_month_query(recent_from, this_month, False))
    claim_q = PREFIX + ("SELECT ?e (COUNT(DISTINCT ?c) AS ?claims) WHERE {\n"
                        '  ?rev a sc:ClaimReview ; sc:inLanguage ?lang ; sc:itemReviewed ?c . FILTER (STR(?lang) = "English")\n'
                        "  ?c sc:mentions ?e .\n} GROUP BY ?e ORDER BY DESC(?claims)")
    cl = run("claim-level entities of the English fact-checks", claim_q)
    names = [p.strip() for p in args.publishers.split(",") if p.strip()]
    values = " ".join(json.dumps(n) for n in names)
    pub_q = PREFIX + ("SELECT ?name (MAX(SUBSTR(STR(?d), 1, 7)) AS ?last) (COUNT(DISTINCT ?rev) AS ?n) WHERE {\n"
                      f"  VALUES ?name {{ {values} }}\n"
                      "  ?a sc:name ?nm . FILTER (STR(?nm) = ?name)\n"
                      "  ?rev a sc:ClaimReview ; sc:author ?a ; sc:datePublished ?d .\n} GROUP BY ?name ORDER BY ?name")
    pubs = run("last month of publication per publisher name", pub_q)

    # The facts the paper quotes.
    n = lambda r, k: int(float(r[k]))  # noqa: E731
    zero_eng = [r["m"] for r in eng if n(r, "n") > 0 and n(r, "withEntities") == 0]
    first_zero = zero_eng[0] if zero_eng and all(n(r, "withEntities") == 0 for r in eng if r["m"] >= zero_eng[0]) else ""
    zero_all = [r["m"] for r in allm if n(r, "n") > 0 and n(r, "withEntities") == 0]
    all_zero = bool(allm) and len(zero_all) == sum(1 for r in allm if n(r, "n") > 0)
    macros = {
        "liveCheckDate": f"{now.day} {MONTHS[now.month - 1]} {now.year}",
        "liveEnglishFirstMonthNoEntity": month_words(first_zero),
        "liveRecentFrom": month_words(recent_from), "liveRecentTo": month_words(this_month),
        "liveRecentAllZero": "yes" if all_zero else "no",
        "liveRecentFactChecks": sum(n(r, "n") for r in allm),
        "liveRecentSentence": (f"no fact-check of any language published from {month_words(recent_from)} to {month_words(this_month)} "
                               f"({sum(n(r, 'n') for r in allm):,} in all) had an entity in its explanation") if all_zero else
                              (f"some fact-checks published from {month_words(recent_from)} to {month_words(this_month)} "
                               f"had entities in their explanation again"),
        "liveDistinctClaimEntities": len(cl),
        "liveClaimEntityTop": cl[0]["e"].rsplit("/", 1)[-1].replace("_", " ") if cl else "none",
    }
    for r in pubs:
        macros["liveLast" + "".join(ch for ch in r["name"].title() if ch.isalpha())] = month_words(r["last"])
    lines = ["% generated by paper-code/stats/live_checks.py on " + now.isoformat(timespec="seconds") + "; do not edit by hand"]
    lines += [tex_macro(k, v) for k, v in macros.items()]
    (out / "live_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report / "live_checks.json", {"endpoint": args.endpoint, "asked": now.isoformat(timespec="seconds"),
                                             "macros": macros, "checks": checks})
    md = [f"# Live checks of {args.endpoint}, asked {now.isoformat(timespec='seconds')}", ""]
    for c in checks:
        md += [f"## {c['name']}", "", "```", c["query"], "```", ""]
        if c["rows"]:
            keys = list(c["rows"][0].keys())
            md += ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
            md += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in c["rows"][:30]]
            if len(c["rows"]) > 30:
                md.append(f"... {len(c['rows']) - 30} more rows in live_checks.json")
        md.append("")
    md += ["## Macros", ""] + [f"- \\{k} = {v}" for k, v in macros.items()]
    (report / "live_checks.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"live_numbers.tex -> {out}; report -> {report / 'live_checks.md'}")


if __name__ == "__main__":
    main()
