"""
title_check.py -- which titles in a writer's sheet are not the page of one concept on
English Wikipedia, and the feedback file the writer receives (the titles are sent back to the writers).

Two kinds of title are flagged, both found on Wikipedia alone, never in the archive:
  A  disambiguation page: a page that only lists the meanings of a word ("Democratic Party"
     lists many parties; the American one is "Democratic Party (United States)"); Wikipedia
     marks such pages with the page property 'disambiguation';
  B  not a page: no page of English Wikipedia has this title (after following redirects).
A title that redirects to a disambiguation page is flagged as A.

The feedback file is in Italian, like the writers' instructions, and contains, per flagged
row: the query id, the formula as the writer wrote it, each flagged title with a link to
Wikipedia, and what to do. With --repairs it also lists the formulas that were
corrected for this writer and asks the writer to confirm them (rows of repair_log.csv whose
writer is this sheet's author and whose 'confirm_with_writer' is yes). No archive count,
no other writer's formula, nothing about the systems is ever written into it.

A row whose formula does not parse (a typing slip still to be repaired) is NOT checked: its
titles cannot be read reliably (a missing quote moves where a title ends). The script names
those rows on the screen; repair them in the sheet and run the script again.

Usage:
  python queries/title_check.py --sheet SHEET.xlsx --writer-name NAME --out FEEDBACK.txt
      [--csv FLAGGED.csv] [--repairs repair_log.csv]
      [--redirect-cache wikipedia_redirects.json] [--disambiguation-cache wikipedia_disambiguation.json] [--offline]
Both caches are JSON files that are released with the collection, so the check can be
repeated offline (--offline: only the caches are used, nothing is asked to Wikipedia).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.boolean import FormulaError, parse, quoted_to_keys  # noqa: E402
from queries.import_formulas import WIKI_PAUSE, RedirectResolver, read_sheet  # noqa: E402



def titles_of(formula: str) -> list[str] | None:
    """The quoted titles of a formula, in order of first appearance; None if the formula does not
    parse with the same parser the import and the systems use (common/boolean.py)."""
    try:
        key_form, keys = quoted_to_keys(formula)
        parse(key_form)
    except FormulaError:
        return None
    return [" ".join(t.split()) for t in keys.values()] or None


class DisambiguationChecker:
    """title -> True if Wikipedia marks the page as a disambiguation page; cached in a JSON file."""

    def __init__(self, cache_path: Path | None, online: bool):
        self.cache_path, self.online = cache_path, online
        self.cache: dict[str, bool] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path and cache_path.exists() else {}

    def check(self, titles: list[str]) -> None:
        todo = sorted({t for t in titles if t not in self.cache})
        if not todo or not self.online:
            return
        print(f"asking Wikipedia whether {len(todo)} pages are disambiguation pages", file=sys.stderr)
        for i in range(0, len(todo), 50):
            chunk = todo[i:i + 50]
            q = RedirectResolver._ask({"action": "query", "titles": "|".join(chunk), "prop": "pageprops",
                                       "ppprop": "disambiguation", "redirects": 1, "format": "json", "formatversion": 2})
            back = {}
            for n in q.get("normalized", []):
                back[n["to"]] = n["from"]
            for r in q.get("redirects", []):
                back[r["to"]] = back.get(r["from"], r["from"])
            dis = {back.get(p["title"], p["title"]) for p in q.get("pages", []) if "disambiguation" in (p.get("pageprops") or {})}
            for t in chunk:
                self.cache[t] = t in dis
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            time.sleep(WIKI_PAUSE)

    def get(self, title: str) -> bool | None:
        return self.cache.get(title)


def wiki_link(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="()_,'-.:")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--writer-name", required=True, help="the writer's first name, used in the file's title")
    ap.add_argument("--out", required=True, help="the feedback file for the writer (.txt, Italian)")
    ap.add_argument("--csv", default=None, help="one line per flagged title (for the person who runs the check, and for the release)")
    ap.add_argument("--repairs", default=None, help="repair_log.csv: corrected formulas to show the writer for confirmation")
    ap.add_argument("--redirect-cache", default=None)
    ap.add_argument("--disambiguation-cache", default=None)
    ap.add_argument("--offline", action="store_true", help="use the caches only")
    args = ap.parse_args()

    sheet = Path(args.sheet)
    rows = [r for r in read_sheet(sheet) if (r.get("formula") or "").strip()]
    author = next((r.get("author_id", "").strip() for r in rows if r.get("author_id")), "")
    resolver = RedirectResolver(Path(args.redirect_cache) if args.redirect_cache else None, online=not args.offline)
    checker = DisambiguationChecker(Path(args.disambiguation_cache) if args.disambiguation_cache else None, online=not args.offline)
    per_row, not_parsed = [], []
    for r in rows:
        ts = titles_of(r["formula"])
        if ts is None:
            not_parsed.append(r["query_id"])
        else:
            per_row.append((r, ts))
    all_titles = sorted({t for _, ts in per_row for t in ts})
    resolver.resolve(all_titles)
    resolved = {t: (resolver.get(t) or {}) for t in all_titles}
    checker.check(sorted({(resolved[t].get("resolved") or t) for t in all_titles}))

    flagged = []  # (row, [(title, kind, resolved title)])
    unknown = 0
    for r, ts in per_row:
        hits = []
        for t in ts:
            rec = resolved[t]
            target = rec.get("resolved") or t
            if rec and rec.get("exists") is False:
                hits.append((t, "B", target))
            elif checker.get(target):
                hits.append((t, "A", target))
            elif not rec or checker.get(target) is None:
                unknown += 1
        if hits:
            flagged.append((r, hits))

    repairs = []
    if args.repairs and Path(args.repairs).exists():
        with open(args.repairs, encoding="utf-8-sig", newline="") as fh:
            repairs = [x for x in csv.DictReader(fh) if x.get("writer") == author and (x.get("confirm_with_writer") or "").strip().lower() == "yes"]

    n_a = sum(1 for _, h in flagged for x in h if x[1] == "A")
    n_b = sum(1 for _, h in flagged for x in h if x[1] == "B")
    L = [f"FORMULE DA RIVEDERE - {args.writer_name}",
         f"Foglio: {sheet.name}. Controllo fatto il {dt.date.today().isoformat()} solo su Wikipedia in inglese.",
         "",
         "COSA TI CHIEDIAMO",
         "Nelle righe elencate sotto, alcuni titoli non sono la pagina di un concetto:",
         "  A. PAGINA DI DISAMBIGUAZIONE: una pagina di Wikipedia che elenca i diversi significati di una parola.",
         "     Esempio: \"Democratic Party\" elenca molti partiti; quello americano è \"Democratic Party (United States)\".",
         "  B. NON È UNA PAGINA: nessuna pagina di Wikipedia in inglese ha questo titolo.",
         "Per ogni titolo elencato: apri il link, scegli la pagina del significato che intendevi e metti il suo titolo",
         "nella formula al posto di quello vecchio. Se nessuna pagina va bene, togli il titolo dalla formula e scrivi",
         "il concetto nella colonna not_expressible.",
         "Scrivi la formula nuova nella colonna \"formula\" della stessa riga del foglio. Non cambiare le altre righe.",
         "",
         f"Righe da rivedere: {len(flagged)} (titoli A: {n_a}, titoli B: {n_b}).",
         ""]
    for r, hits in flagged:
        L += [f"--- {r['query_id']}", "Formula: " + " ".join(r["formula"].split())]
        for t, kind, target in hits:
            if kind == "A":
                via = f" (porta a \"{target}\")" if target != t else ""
                L.append(f"  [A] \"{t}\"{via} è una pagina di disambiguazione: {wiki_link(target)}")
            else:
                L.append(f"  [B] \"{t}\" non è una pagina di Wikipedia in inglese")
        L.append("")
    if repairs:
        L += ["FORMULE CORRETTE DA NOI (solo da controllare)",
              "In queste righe abbiamo corretto la formula come scritto sotto. Se intendevi un'altra cosa,",
              "scrivi la formula giusta nella colonna \"formula\" della stessa riga.", ""]
        for x in repairs:
            L += [f"--- {x['query_id']}", "Prima: " + " ".join(x["original"].split()), "Dopo:  " + " ".join(x["repaired"].split()),
                  "Perché: " + x.get("reason_it", "").strip(), ""]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    if args.csv:
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["query_id", "author_id", "title_as_written", "kind", "wikipedia_target"])
            for r, hits in flagged:
                for t, kind, target in hits:
                    w.writerow([r["query_id"], author, t, "disambiguation" if kind == "A" else "not a page", target])
    print(f"{args.writer_name}: {len(rows)} formulas, {len(all_titles)} distinct titles; flagged rows {len(flagged)}, "
          f"disambiguation titles {n_a}, not a page {n_b}; titles not checked (offline, not in the caches) {unknown}; "
          f"repairs to confirm {len(repairs)} -> {args.out}")
    if not_parsed:
        print(f"NOT CHECKED, the formula does not parse (repair the row, then run again): {', '.join(not_parsed)}")


if __name__ == "__main__":
    main()
