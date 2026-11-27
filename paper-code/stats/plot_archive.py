"""
plot_archive.py -- the figures of the paper's data section and the LaTeX
macros for every number it quotes, from the files that stats/archive_stats.py
and snapshot/build_store.py wrote. Nothing is typed by hand: if the store is
rebuilt, run archive_stats.py and then this script, and the paper's numbers
and figures follow.

Inputs (all produced by the pipeline):
  WORK/stats/archive_stats.json, per_month.csv, country_year.csv, country_month.csv, publishers.csv
  WORK/stats/publishers_map.csv               (snapshot/publisher_map.py; optional: merges the graph's
                                               duplicate publisher names into one organisation)
  WORK/corpus.jsonl, WORK/entities_cr.jsonl   (run_all.py --steps prepare --node cr)
  <store parent>/snapshot_manifest.json        (snapshot/build_store.py)
Outputs (--out, default latex/figures next to the repository root):
  per_month.pdf        English fact-checks per month since 2016, total and the three largest countries
  country_year.pdf     stacked bars per year: United States, India, United Kingdom, other countries
  language_rule.pdf    what the language rule removed, by reason and by detected language
  entities_per_doc.pdf how many explanation-level entities a document carries (documents with at least one)
  entity_coverage.pdf  per month, the share of documents with an explanation text and with an explanation-level entity
  publisher_spans.pdf  first and last month of every organisation with at least 100 documents (dead feeds);
                       the graph's names of one organisation (snopescom and snopes.com) are merged
  numbers.tex          \\newcommand macros (\\nDocs, \\nNotEnglish, ...) for the LaTeX source
  figures.json         the numbers behind every figure, for checking

Colours: the validated categorical order of the dataviz palette (blue, orange,
aqua, yellow), one blue for single-series charts. At most four series per
chart; further countries are folded into "other".

Usage:
  python stats/plot_archive.py --work WORK --store STORE [--out DIR] [--since 2016-01]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonl import read_jsonl, write_json  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY = "#52514e"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]
TOP_COUNTRIES = 3  # the rest is folded into "other"
COUNTRY_MIN_DOCS = 50   # a country is counted (\nCountries) when it has at least this many documents
SPANS_MIN_DOCS = 100    # an organisation gets a bar in publisher_spans.pdf when it has at least this many documents
ENTITIES_CAP = 100      # the last bar of entities_per_doc.pdf collects the documents with this many entities or more
# A feed is "stopped" when the organisation has at least DEAD_MIN_DOCS documents and its last month is more than
# DEAD_GAP_MONTHS before the last month of the archive. Display names only for the text; the figure keeps the IFCN names.
DEAD_MIN_DOCS, DEAD_GAP_MONTHS = 500, 12
# The organisations' names as the text writes them, keyed by their IFCN name (publishers_map.csv). Names only, no numbers.
DISPLAY = {"politifact": "PolitiFact", "factcheck.org": "FactCheck.org", "the-washington-post-fact-checker": "the Washington Post",
           "india-today-fact-check": "India Today", "check-your-fact": "Check Your Fact", "thip-healthtech-pvt-ltd": "THIP Media",
           "newsmeter-fifth-estate-digital-private-limited": "Newsmeter", "snopes.com": "Snopes",
           "factspace-west-africa-(ghanafact)": "GhanaFact", "full-fact": "Full Fact", "aap-factcheck": "AAP",
           "lead-stories": "Lead Stories", "africa-check": "Africa Check", "science-feedback": "Science Feedback",
           "facta": "Facta", "logically-facts": "Logically Facts", "vishvas-news": "Vishvas News", "boom": "Boom",
           "alt-news": "Alt News", "the-quint": "The Quint", "factcrescendo": "Fact Crescendo",
           "lighthousejournalism.com": "Lighthouse Journalism", "annie-lab": "Annie Lab",
           "factcheckhub[international-centre-for-investigative-reporting]": "FactCheckHub", "digiteye": "Digiteye",
           "medical-dialogues": "Medical Dialogues", "ferret-fact-service": "Ferret Fact Service", "factcheckni": "FactCheckNI",
           "fact-check-zimbabwe": "ZimFact", "the-conversation-factcheck": "The Conversation", "afp-fact-check": "AFP Fact Check",
           "factly-media-&-research": "Factly", "dubawa": "Dubawa"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def month_words(m: str) -> str:
    """'2025-04' -> 'April 2025'."""
    return f"{MONTHS[int(m[5:7]) - 1]} {m[:4]}" if m else "none"


def entity_stop(n_by_month: dict, e_by_month: dict) -> str:
    """The first month of the final run of months in which no document has an explanation-level entity, if that run
    is at least three months long and reaches the archive's last month; '' when entity linking never stopped."""
    months = sorted(m for m in n_by_month if n_by_month[m] > 0)
    stop = ""
    for m in reversed(months):
        if e_by_month.get(m, 0) == 0:
            stop = m
        else:
            break
    return stop if stop and sum(1 for m in months if m >= stop) >= 3 else ""


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def style(plt):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GREY, "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def save(fig, path):
    """Save a figure as PDF without the creation time, so that the same data always gives the same file
    (reproduce_paper.py compares the files byte by byte)."""
    fig.savefig(path, metadata={"CreationDate": None})


def tex_macro(name: str, value) -> str:
    """A LaTeX macro with the number formatted with thousands separators (or the text as given)."""
    if isinstance(value, float):
        text = f"{value:.1f}"
    elif isinstance(value, int):
        text = f"{value:,}"
    else:
        text = str(value)
    return f"\\newcommand{{\\{name}}}{{{text}}}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--store", required=True, help="the store; snapshot_manifest.json is read from its parent folder")
    ap.add_argument("--out", default=None, help="default: <repository>/latex/figures")
    ap.add_argument("--since", default="2016-01", help="first month of the time charts")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    work = Path(args.work)
    stats = work / "stats"
    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent.parent / "latex" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    rep = json.loads((stats / "archive_stats.json").read_text(encoding="utf-8"))
    manifest_path = Path(args.store).resolve().parent / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    behind = {}

    # ---- Figure 1: documents per month, total and the three largest countries ----
    per_month = {r["month"]: int(r["fact_checks"]) for r in read_csv(stats / "per_month.csv") if r["month"] != "missing"}
    months = sorted(m for m in per_month if m >= args.since)
    cm = defaultdict(dict)
    for r in read_csv(stats / "country_month.csv"):
        if r["month"] != "missing":
            cm[r["country"]][r["month"]] = int(r["fact_checks"])
    countries = [c for c, _ in rep.get("per_country", [])][:TOP_COUNTRIES]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x = list(range(len(months)))
    ax.plot(x, [per_month[m] for m in months], color=GREY, linewidth=2, label="all countries")
    for i, c in enumerate(countries):
        ax.plot(x, [cm[c].get(m, 0) for m in months], color=SERIES[i], linewidth=1.6, label=c)
    ticks = [i for i, m in enumerate(months) if m.endswith("-01")]
    ax.set_xticks(ticks)
    ax.set_xticklabels([months[i][:4] for i in ticks])
    ax.set_ylabel("fact-checks per month")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    save(fig, out / "per_month.pdf")
    plt.close(fig)
    behind["per_month"] = {"months": months, "total": [per_month[m] for m in months], **{c: [cm[c].get(m, 0) for m in months] for c in countries}}

    # ---- Figure 2: country x year, stacked ----
    cy = defaultdict(dict)
    for r in read_csv(stats / "country_year.csv"):
        if r["year"] != "missing":
            cy[r["country"]][r["year"]] = int(r["fact_checks"])
    years = sorted({y for c in cy for y in cy[c] if y >= args.since[:4]})
    others = [c for c in cy if c not in countries]
    series = {c: [cy[c].get(y, 0) for y in years] for c in countries}
    series["other countries"] = [sum(cy[c].get(y, 0) for c in others) for y in years]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    bottom = [0] * len(years)
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(years, vals, bottom=bottom, color=SERIES[i], width=0.72, label=name, edgecolor="white", linewidth=0.8)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("fact-checks per year")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    save(fig, out / "country_year.pdf")
    plt.close(fig)
    behind["country_year"] = {"years": years, **series}

    # ---- Figure 3: the language rule, by reason and by language ----
    lr = rep.get("language_rule") or manifest.get("language_rule") or {}
    reasons = list(lr.get("by_reason", {}).items())
    langs = list(lr.get("by_language", {}).items())[:10]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw={"width_ratios": [1.1, 1]})
    for ax, items, title in ((a1, reasons, "by reason"), (a2, langs, "by detected language (conditions 4 to 6)")):
        names = [k.replace("_", " ").replace("article", "explanation") for k, _ in items][::-1]  # the store's reason codes say article
        vals = [v for _, v in items][::-1]
        ax.barh(names, vals, color=BLUE, height=0.65)
        for y, v in enumerate(vals):
            ax.text(v, y, f" {v:,}", va="center", fontsize=7.5, color=GREY)
        ax.set_title(title, fontsize=9, color=GREY, loc="left")
        ax.set_xlim(0, max(vals or [1]) * 1.25)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, out / "language_rule.pdf")
    plt.close(fig)
    behind["language_rule"] = {"by_reason": dict(reasons), "by_language": dict(langs)}

    # ---- Figure 4: entities per document on the ClaimReview node ----
    counts = Counter()
    ent_path = work / "entities_cr.jsonl"
    if ent_path.exists():
        for r in read_jsonl(ent_path):
            counts[min(len(r["entities"]), ENTITIES_CAP)] += 1
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ks = sorted(counts)
    ax.bar(ks, [counts[k] for k in ks], color=BLUE, width=0.9)
    ax.set_xlabel("entities on the fact-check explanation (100 = 100 or more)")
    ax.set_ylabel("documents")
    save(fig, out / "entities_per_doc.pdf")
    plt.close(fig)
    behind["entities_per_doc"] = {str(k): counts[k] for k in ks}

    # ---- Figure 5: per month, the share of documents with an explanation text and with an explanation-level entity ----
    with_ent = set()
    if ent_path.exists():
        with_ent = {r["cr_iri"] for r in read_jsonl(ent_path) if r["entities"]}
    corpus_path = work / "corpus.jsonl"
    docs = list(read_jsonl(corpus_path)) if corpus_path.exists() else []
    n_m, e_m, a_m = Counter(), Counter(), Counter()
    for d in docs:
        m = (d.get("date_published") or "")[:7]
        if m:
            n_m[m] += 1
            e_m[m] += d["cr_iri"] in with_ent
            a_m[m] += bool((d.get("article_text") or "").strip())
    stop = entity_stop(n_m, e_m)
    cmonths = [m for m in sorted(n_m) if m >= args.since]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    x = list(range(len(cmonths)))
    share_a = [100.0 * a_m[m] / n_m[m] for m in cmonths]
    share_e = [100.0 * e_m[m] / n_m[m] for m in cmonths]
    ax.plot(x, share_a, color=GREY, linewidth=1.6, label="with an explanation text")
    ax.plot(x, share_e, color=BLUE, linewidth=2, label="with at least one entity in the explanation")
    ticks = [i for i, m in enumerate(cmonths) if m.endswith("-01")]
    ax.set_xticks(ticks)
    ax.set_xticklabels([cmonths[i][:4] for i in ticks])
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of the month's fact-checks")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    save(fig, out / "entity_coverage.pdf")
    plt.close(fig)
    behind["entity_coverage"] = {"months": cmonths, "documents": [n_m[m] for m in cmonths],
                                 "with_article_pct": share_a, "with_entity_pct": share_e, "entity_stop_month": stop}

    # ---- Figure 6: organisations' first and last month (the graph's duplicate names of one organisation merged) ----
    rows, seen = [], set()
    for r in read_csv(stats / "publishers.csv"):
        if r["publisher"] not in seen:  # publishers.csv can list one name twice (two publisher nodes with one name)
            seen.add(r["publisher"])
            rows.append(r)
    map_path = stats / "publishers_map.csv"
    org_of = {}
    if map_path.exists():
        org_of = {r["publisher"]: (r.get("ifcn_name") or r["publisher"]) for r in read_csv(map_path)}
    merged = {}
    for r in rows:
        org = org_of.get(r["publisher"], r["publisher"])
        g = merged.setdefault(org, {"publisher": org, "first_month": "", "last_month": "", "fact_checks": 0, "names": []})
        g["fact_checks"] += int(r["fact_checks"])
        g["names"].append(r["publisher"])
        if r["first_month"]:
            g["first_month"] = min(g["first_month"] or r["first_month"], r["first_month"])
            g["last_month"] = max(g["last_month"], r["last_month"])
    pubs = [g for g in merged.values() if g["fact_checks"] >= SPANS_MIN_DOCS and g["first_month"]]
    pubs.sort(key=lambda r: r["last_month"])

    def mnum(m):  # 'yyyy-mm' -> months since year 0, for a numeric axis
        return int(m[:4]) * 12 + int(m[5:7]) - 1
    last_month = max(per_month) if per_month else "2026-01"
    fig, ax = plt.subplots(figsize=(6.4, 0.22 * len(pubs) + 0.8))
    for i, r in enumerate(pubs):
        a, b = mnum(max(r["first_month"], args.since)), mnum(r["last_month"])
        ax.barh(i, max(b - a, 1), left=a, color=BLUE, height=0.55)
    ax.axvline(mnum(last_month), color=ORANGE, linewidth=1.2)
    ax.text(mnum(last_month), len(pubs) - 0.4, " snapshot", color=ORANGE, fontsize=7, va="top")
    yrs = list(range(int(args.since[:4]), int(last_month[:4]) + 1, 2))
    ax.set_xticks([y * 12 for y in yrs])
    ax.set_xticklabels([str(y) for y in yrs])
    ax.set_yticks(range(len(pubs)))
    ax.set_yticklabels([f"{r['publisher'][:30]} ({int(r['fact_checks']):,})" for r in pubs], fontsize=6.5)
    ax.set_xlim(mnum(args.since) - 3, mnum(last_month) + 14)
    ax.grid(axis="y", visible=False)
    save(fig, out / "publisher_spans.pdf")
    plt.close(fig)
    behind["publisher_spans"] = [(r["publisher"], r["first_month"], r["last_month"], int(r["fact_checks"]), r["names"]) for r in pubs]

    def months_between(a, b):
        return (int(b[:4]) - int(a[:4])) * 12 + int(b[5:7]) - int(a[5:7])
    dead = [g for g in pubs if g["fact_checks"] >= DEAD_MIN_DOCS and months_between(g["last_month"], last_month) > DEAD_GAP_MONTHS]
    dead.sort(key=lambda g: g["last_month"])
    renamed = [g for g in pubs if len(g["names"]) > 1]
    behind["stopped_feeds"] = [(g["publisher"], g["last_month"], g["fact_checks"]) for g in dead]
    behind["organisations_with_several_names"] = [(g["publisher"], g["names"]) for g in renamed]

    # ---- numbers.tex: every number the data section quotes ----
    cov = rep["entity_coverage"]
    ecr = rep["entities_per_cr"]
    per_country = dict(rep.get("per_country", []))
    n = rep["n_documents"]
    removed = manifest.get("removed", {})
    macros = {
        "nDocs": n, "nLoaded": manifest.get("claimreviews_loaded", 0), "nScrapingFailures": removed.get("scraping_failures", 0),
        "nNoClaimText": removed.get("no_claim_text", 0), "nLanguageInput": lr.get("documents", 0), "nNotEnglish": lr.get("not_english", 0),
        "pctNotEnglish": 100.0 * lr.get("not_english", 0) / lr.get("documents", 1) if lr else 0.0,
        "nPublishers": rep["n_publishers"], "nCountries": sum(1 for c, v in per_country.items() if v >= COUNTRY_MIN_DOCS),
        "yearMin": str(rep["year_min_median_max"][0]), "yearMedian": str(rep["year_min_median_max"][1]), "yearMax": str(rep["year_min_median_max"][2]),
        "nUndated": sum(v for _, v in rep["undated_by_publisher"]), "nWithArticle": rep["article_text_present"],
        "pctWithArticle": 100.0 * rep["article_text_present"] / n,
        "nWithCrEntities": cov["cr_node_any"], "pctWithCrEntities": 100.0 * cov["cr_node_any"] / n,
        "nNoCrEntity": cov["neither"] + cov["only_claim"], "pctNoCrEntity": 100.0 * (cov["neither"] + cov["only_claim"]) / n,
        "nNoEntityEither": cov["neither"], "pctNoEntityEither": 100.0 * cov["neither"] / n,
        "medianEntities": int(ecr["median"]) if ecr["median"] is not None else 0, "meanEntities": float(ecr["mean"] or 0), "maxEntities": ecr["max"] or 0,
        "nAtLeastTwoEntities": ecr["at_least_2"], "nClaimNodeMars": cov["claim_node_any"],
        "nSharedClaim": rep["shared_claim_nodes"]["documents_sharing_a_claim_node"], "nSharedClaimGroups": rep["shared_claim_nodes"]["distinct_shared_claim_nodes"],
        "snapshotBuilt": manifest.get("built", "")[:10], "mirrorSha": manifest.get("mirror_sha256", "")[:12],
    }
    for i, (c, v) in enumerate(rep.get("per_country", [])[:TOP_COUNTRIES]):
        macros[f"country{'ABC'[i]}Name"] = c
        macros[f"country{'ABC'[i]}N"] = v
        macros[f"country{'ABC'[i]}Pct"] = 100.0 * v / n
    for key, val in lr.get("by_reason", {}).items():
        macros["reason" + "".join(w.capitalize() for w in key.split("_"))] = val
    for key, val in list(lr.get("by_language", {}).items())[:6]:
        macros["lang" + key.capitalize()] = val
    cal = (lr.get("calibration") or {}).get("false_alarms_by_claim_length", {})
    for label, name in (("under 30", "calibShort"), ("30 to 59", "calibMid"), ("60 and more", "calibLong")):
        c = cal.get(label, {})
        macros[name + "Rate"] = f"{c['rate_percent']:.2f}" if c.get("rate_percent") is not None else "n/a"
        macros[name + "N"] = c.get("claims", 0)
    top = rep["top_entities"]["cr_node"]
    macros["topCrEntities"] = ", ".join(f"{e.rsplit('/', 1)[-1].replace('_', ' ')} ({v:,})" for e, v in top[:5])
    # The documents without an explanation-level entity, split: no explanation text at all; an explanation text, published
    # before the month entity linking stopped (or undated); an explanation text, published from that month on.
    if docs:
        has_art = lambda d: bool((d.get("article_text") or "").strip())  # noqa: E731
        after = lambda d: bool(stop) and (d.get("date_published") or "")[:7] >= stop  # noqa: E731
        no_ent = [d for d in docs if d["cr_iri"] not in with_ent]
        art_before = [d for d in docs if has_art(d) and not after(d)]
        macros.update({
            "entityStopMonth": month_words(stop),
            "nAfterEntityStop": sum(1 for d in docs if after(d)),
            "pctAfterEntityStop": 100.0 * sum(1 for d in docs if after(d)) / n,
            "nNoCrEntityNoArticle": sum(1 for d in no_ent if not has_art(d)),
            "nNoCrEntityArticleAfterStop": sum(1 for d in no_ent if has_art(d) and after(d)),
            "nNoCrEntityArticleBeforeStop": sum(1 for d in no_ent if has_art(d) and not after(d)),
            "pctEntityBeforeStop": 100.0 * sum(1 for d in art_before if d["cr_iri"] in with_ent) / max(len(art_before), 1),
        })
    feeds = [f"{DISPLAY.get(g['publisher'], g['publisher'])} ({month_words(g['last_month'])})" for g in dead]
    macros["stoppedFeeds"] = ("none" if not feeds else " and ".join(feeds) if len(feeds) <= 2 else ", ".join(feeds[:-1]) + ", and " + feeds[-1])
    macros["nStoppedFeeds"] = len(dead)
    macros["archiveLastMonth"] = month_words(last_month)
    macros.update({"countryMinDocs": COUNTRY_MIN_DOCS, "spansMinDocs": SPANS_MIN_DOCS, "entitiesCap": ENTITIES_CAP,
                   "figuresSinceYear": args.since[:4], "stoppedMinDocs": DEAD_MIN_DOCS, "stoppedGapMonths": DEAD_GAP_MONTHS})
    lines = ["% generated by paper-code/stats/plot_archive.py; do not edit by hand"] + [tex_macro(k, v) for k, v in macros.items()]
    (out / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "figures.json", {"behind_the_figures": behind, "macros": macros})
    print(f"figures and numbers.tex -> {out} ({len(macros)} macros)")


if __name__ == "__main__":
    main()
