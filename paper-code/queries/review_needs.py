"""
review_needs.py -- the human review of the merged candidates, under the rules
of queries/REVIEW-RULES.md, with every decision logged, and the blind
re-review of a random sample by a second person.

  sheet      write the review workbook (one row per cluster of every cell):
             cluster_id, cell, cells (all the cells its members came from),
             period, country_fit (the model's own opinion, for rule D2),
             mentions, topic, claim, spreader, verdict_words, verdict_found,
             passage, words_in_passage (the percentage of the statement's
             content words found in the passage, shaded orange under 20 and
             yellow from 20 to 39: a help for rule D4, never a decision),
             sources, decision (K or D),
             rule (D1-D5), note
             QUOTA MODE (with --allocation): the rows of
             every cell are put in a seeded random order (column "order", 1 =
             review first); every cell shows its target_K = its number of
             queries in the allocation + --margin (default 2), at most its
             number of rows, plus, in a country whose years with no row left
             are short of queries, its share of that shortfall (the draw tops
             up short years from the other years: extended_targets); K_so_far counts the K written so far in the cell
             (an Excel formula) and "status" says DONE when the target is
             reached (the done rows turn grey). The reviewer reviews each cell
             from its top and moves to the next cell at DONE. --carry OLD.xlsx
             copies the decisions already made in an earlier workbook, matched
             by cluster_id.
  import     read the filled workbook back into WORK/drafting/reviewed.jsonl
             (every decided row needs K or D; a D needs a rule; a D3 row
             that the draw uses needs, in its note, the cluster_id of a kept
             twin of any country or year that the draw uses too:
             d3_twin_errors) and write review.md. Without quota mode every row must be decided. In quota
             mode an empty decision is allowed: per cell, the reviewed prefix is
             the run of decided rows from the top of the random order, and only
             the rows inside it are eligible for the draw ("eligible": true).
             This is inverse sampling (Haldane 1945): the first K rows of a
             random order are a random sample of all the K rows of the cell,
             so a random draw from them is a random draw from all the K rows.
             A decided row below an undecided one is logged but not eligible.
             WORK/drafting/review_status.json says per cell: target reached,
             all rows reviewed, or continue (draw_queries.py refuses to draw
             while a cell says continue).
  second     a blind workbook with a seeded random --fraction of the decided
             rows, without the first decisions, for the second reviewer
  agreement  Cohen's kappa between the first and the second reviewer on the
             sample, written to WORK/drafting/review_agreement.md

Usage:
  python queries/review_needs.py sheet --work WORK --out review.xlsx
  python queries/review_needs.py sheet --work WORK --out review_quota.xlsx --allocation ../latex/figures/allocation.json
         [--margin 2] [--order-seed 20261030] [--carry review.xlsx]
  python queries/review_needs.py import --work WORK --sheet review.xlsx [--reviewer NAME]
  python queries/review_needs.py import --work WORK --sheet review_quota.xlsx --allocation ../latex/figures/allocation.json
  python queries/review_needs.py second --work WORK --out second.xlsx [--fraction 0.2] [--seed 7]
  python queries/review_needs.py agreement --work WORK --sheet second.xlsx
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonl import read_jsonl, write_jsonl  # noqa: E402

COLUMNS = ["cluster_id", "cell", "cells", "period", "country_fit", "mentions", "topic", "claim", "spreader", "verdict_words", "verdict_found",
           "passage", "words_in_passage", "sources", "decision", "rule", "note"]
QUOTA_COLUMNS = ["cluster_id", "cell", "order", "target_K"] + COLUMNS[2:] + ["K_so_far", "status"]
RULES = {"D1", "D2", "D3", "D4", "D5"}
CLUSTER_ID = re.compile(r"[a-z]+(?:-[a-z]+)*-\d{4}:c\d{4}")  # us-2019:c0007, south-africa-2020:c0003
FOLDER = {"United States": "us", "India": "india", "United Kingdom": "uk", "Australia": "australia", "Nigeria": "nigeria",
          "Ghana": "ghana", "South Africa": "south-africa", "Ireland": "ireland", "Hong Kong": "hong-kong",
          "France": "france", "Zimbabwe": "zimbabwe"}  # the same map as draw_queries.py and prepare_cells.py
LOW_MATCH, SOME_MATCH = 20, 40  # percent; on the real run: under 20, 78 of 1,504 rows, mostly an EDMO item's
                                # statement next to the passage of another item (rule D4), shaded orange; 20 to 39, 71
                                # rows, about 3 in 10 of them mismatched in a sample of 10, shaded yellow
STOP = set("the a an of to in and or is are was were be been being that this these those for on with as by at from it its "
           "their they them he she his her not no has have had will would can could who which what about into than".split())


def content_words(s: str) -> list[str]:
    return [w for w in re.findall(r"\w+", unicodedata.normalize("NFKC", s).lower()) if len(w) > 2 and w not in STOP]


def words_in_passage(claim: str, passage: str) -> int:
    """The percentage of the statement's content words (no short or function words) that also appear in its passage,
    a word counting as present when a passage word starts with its first five letters ("attacked" ~ "attacks"). Only
    a help for rule D4: a low value means "read this passage carefully", never a decision."""
    c = content_words(claim)
    if not c:
        return 100
    p = content_words(passage)
    stems = {w[:5] for w in p}
    return round(100 * sum(1 for w in c if w[:5] in stems) / len(c))


def d3_twin_errors(reviewed: list[dict], quota_mode: bool) -> list[str]:
    """Rule D3 ("it says the same as another kept claim of any country or year") needs its kept twin: the note of every
    D3 row that the draw uses must name the cluster_id of a row that is K and, in quota mode, itself used by the draw
    (above the first empty row of its cell). Without this check, two rows that say the same could both end up dropped
    (the twin changed to D later, or left below a gap) and the narrative would leave the pool unnoticed."""
    by_id = {r["cluster_id"]: r for r in reviewed}
    errors = []
    for r in reviewed:
        if r.get("rule") != "D3" or (quota_mode and not r.get("eligible")):
            continue
        twins = [t for t in CLUSTER_ID.findall(r.get("note", "") or "") if t != r["cluster_id"]]
        kept = [t for t in twins if t in by_id and by_id[t].get("decision") == "K" and (by_id[t].get("eligible") or not quota_mode)]
        if not kept:
            why = "names no cluster_id" if not twins else "names no kept row the draw uses (" + ", ".join(
                f"{t}: {by_id[t].get('decision') or 'undecided'}{'' if by_id[t].get('eligible') or not quota_mode else ', below a gap'}"
                if t in by_id else f"{t}: unknown" for t in twins) + ")"
            errors.append(f"{r['cluster_id']}: D3 note {why}")
    return errors


def load_merged(work: Path) -> list[dict]:
    rows = []
    for f in sorted((work / "drafting").glob("merged_*.jsonl")):
        rows.extend(read_jsonl(f))
    return rows


def quotas(allocation: Path) -> dict[str, int]:
    """The number of queries of every cell ("us-2020": 9) from the allocation file."""
    cells = json.load(open(allocation, encoding="utf-8"))["cells"]
    return {f"{FOLDER[country]}-{year}": n for country, years in cells.items() for year, n in years.items()}


def random_order(rows: list[dict], seed: int) -> list[dict]:
    """Every cell's rows in a seeded random order (its own generator per cell, so one cell's order does not depend on
    the others); writes "order" (1 = first to review) and "rows_in_cell"."""
    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)
    out = []
    for cell in sorted(by_cell):
        group = sorted(by_cell[cell], key=lambda r: r["cluster_id"])
        random.Random(f"review-order:{seed}:{cell}").shuffle(group)
        for i, r in enumerate(group, start=1):
            r["order"], r["rows_in_cell"] = i, len(group)
        out += group
    return out


def cell_stats(rows: list[dict]) -> dict[str, dict]:
    """Per cell, from rows carrying "order" and "decision": its rows, the reviewed prefix (decided rows from the top of
    the random order, no gap) and the K inside it."""
    out = {}
    for cell in sorted({r["cell"] for r in rows}):
        group = sorted((r for r in rows if r["cell"] == cell), key=lambda r: r["order"])
        prefix = 0
        while prefix < len(group) and group[prefix].get("decision", "") in ("K", "D"):
            prefix += 1
        out[cell] = {"rows": len(group), "reviewed_from_top": prefix, "K_from_top": sum(1 for r in group[:prefix] if r["decision"] == "K")}
    return out


def extended_targets(stats: dict[str, dict], quota: dict[str, int], margin: int) -> dict[str, int]:
    """The K every cell must reach. Base: its queries + margin, at most its rows. Top-up: the draw fills a
    short year from the K that its country's other years leave (draw_queries.py --pool-short), but in quota mode those
    years stop at queries + margin. So the shortfall of the country's closed cells (no row left to review; an allocated
    cell with no row at all counts as closed with 0 K), net of the spare K those closed cells already hold, is added to
    the targets of the country's cells that still have rows, in proportion to their rows (largest remainder); every
    cell keeps its own margin free for the pilot."""
    allc = {c: stats.get(c, {"rows": 0, "reviewed_from_top": 0, "K_from_top": 0}) for c in set(stats) | set(quota)}
    target = {c: min(quota.get(c, 0) + margin, s["rows"]) for c, s in allc.items()}
    by_country: dict[str, list[str]] = {}
    for c in sorted(allc):
        by_country.setdefault(c.rsplit("-", 1)[0], []).append(c)
    for cells in by_country.values():
        closed = [c for c in cells if allc[c]["reviewed_from_top"] == allc[c]["rows"]]
        open_ = [c for c in cells if c not in closed]
        extra = (sum(max(0, quota.get(c, 0) - allc[c]["K_from_top"]) for c in closed)
                 - sum(max(0, allc[c]["K_from_top"] - quota.get(c, 0)) for c in closed))
        if extra <= 0 or not open_:
            continue
        rows = sum(allc[c]["rows"] for c in open_)
        share = {c: extra * allc[c]["rows"] / rows for c in open_}
        add = {c: int(share[c]) for c in open_}
        for c in sorted(open_, key=lambda c: (add[c] - share[c], c))[:extra - sum(add.values())]:
            add[c] += 1
        for c in open_:
            target[c] = min(quota.get(c, 0) + margin + add[c], allc[c]["rows"])
    return {c: t for c, t in target.items() if c in stats}


def write_xlsx(path: Path, rows: list[dict], columns: list[str]) -> None:
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "review"
    ws.append(columns)
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in rows:
        ws.append([r.get(c, "") for c in columns])
    widths = {"cluster_id": 24, "cell": 14, "cells": 16, "period": 12, "country_fit": 9, "mentions": 9, "topic": 14, "need": 60, "claim": 55, "spreader": 22, "verdict_words": 30,
              "verdict_found": 9, "passage": 60, "words_in_passage": 10, "sources": 30, "decision": 9, "rule": 7, "note": 30,
              "order": 7, "target_K": 9, "K_so_far": 9, "status": 24}
    for i, c in enumerate(columns, start=1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = widths.get(c, 15)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    if "words_in_passage" in columns:  # a low match is shaded so that it catches the eye; it decides nothing
        low, some = PatternFill("solid", fgColor="F8CBAD"), PatternFill("solid", fgColor="FFF2CC")
        wi = columns.index("words_in_passage") + 1
        for r in range(2, len(rows) + 2):
            v = ws.cell(row=r, column=wi).value
            if isinstance(v, int) and v < SOME_MATCH:
                ws.cell(row=r, column=wi).fill = low if v < LOW_MATCH else some
    if "decision" in columns:
        dv = DataValidation(type="list", formula1='"K,D"', allow_blank=True)
        dv2 = DataValidation(type="list", formula1='"D1,D2,D3,D4,D5"', allow_blank=True)
        ws.add_data_validation(dv)
        ws.add_data_validation(dv2)
        ci = columns.index("decision") + 1
        ri = columns.index("rule") + 1
        for r in range(2, len(rows) + 2):
            dv.add(ws.cell(row=r, column=ci))
            dv2.add(ws.cell(row=r, column=ri))
    if "K_so_far" in columns:  # quota mode: a running count of the K of the row's cell and a stop signal, as Excel formulas
        from openpyxl.formatting.rule import FormulaRule
        from openpyxl.workbook.properties import CalcProperties
        col = {c: openpyxl.utils.get_column_letter(columns.index(c) + 1) for c in ("cell", "order", "target_K", "decision", "K_so_far", "status")}
        block: dict[str, list[int]] = {}  # the sheet rows of every cell (the rows are sorted by cell, then by order)
        for i, r in enumerate(rows, start=2):
            block.setdefault(r["cell"], [i, i])[1] = i
        dc, oc, tc, kc = col["decision"], col["order"], col["target_K"], col["K_so_far"]
        for i, r in enumerate(rows, start=2):
            a, b = block[r["cell"]]
            rng_ = f"${dc}${a}:${dc}${b}"
            first_empty = f"IFERROR(MATCH(TRUE,INDEX({rng_}=\"\",0),0),0)"  # position of the cell's first empty decision, 0 if none
            limit = f"IF({first_empty}=0,{oc}{i},MIN({oc}{i},{first_empty}-1))"  # count only the rows from the top with no gap
            # K_so_far = the K from the top of the cell down to this row, stopping at the first empty row: the import's rule
            ws[f"{kc}{i}"] = f"=IF({limit}<1,0,COUNTIF(${dc}${a}:INDEX({rng_},{limit}),\"K\"))"
            ws[f"{col['status']}{i}"] = (f"=IF({kc}{i}>={tc}{i},\"DONE: go to the next cell\",IF({oc}{i}={first_empty},\"<< review from here\","
                                         f"IF(AND({first_empty}=0,{oc}{i}={r['rows_in_cell']}),\"all rows reviewed\",\"\")))")
        last = openpyxl.utils.get_column_letter(len(columns))
        sc = col["status"]
        ws.conditional_formatting.add(f"A2:{last}{len(rows) + 1}", FormulaRule(
            formula=[f'${sc}2="DONE: go to the next cell"'], fill=PatternFill("solid", fgColor="D9D9D9")))
        ws.conditional_formatting.add(f"A2:{last}{len(rows) + 1}", FormulaRule(
            formula=[f'${sc}2="<< review from here"'], fill=PatternFill("solid", fgColor="DDEBF7")))
        wb.calculation = CalcProperties(fullCalcOnLoad=True)
    ws.freeze_panes = "A2"
    wb.save(path)


def read_xlsx(path: Path) -> list[dict]:
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    out = []
    for r in rows[1:]:
        if r is None or all(v is None for v in r):
            continue
        out.append({header[i]: ("" if v is None else str(v).strip()) for i, v in enumerate(r) if i < len(header)})
    return out


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sheet", "import", "second", "agreement"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sheet", type=Path, default=None)
    ap.add_argument("--reviewer", default="reviewer")
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--allocation", type=Path, default=None, help="quota mode: the allocation file (latex/figures/allocation.json)")
    ap.add_argument("--margin", type=int, default=2, help="quota mode: K rows wanted beyond a cell's queries (pilot and second-review margin)")
    ap.add_argument("--order-seed", type=int, default=20261030, help="quota mode: the seed of the random order within every cell")
    ap.add_argument("--carry", type=Path, default=None, help="quota mode: an earlier workbook whose decisions are copied by cluster_id")
    args = ap.parse_args()
    d = args.work / "drafting"

    if args.command == "sheet":
        rows = load_merged(args.work)
        for r in rows:
            r["sources"] = "; ".join(r["docs"])
            r["cells"] = ", ".join(r.get("cells", [r["cell"]]))
            r["words_in_passage"] = words_in_passage(r["claim"], r["passage"])
            r["decision"], r["rule"], r["note"] = "", "", ""
        columns, extra = COLUMNS, ""
        if args.allocation:
            quota = quotas(args.allocation)
            rows = random_order(rows, args.order_seed)
            carried = 0
            if args.carry:
                old = {s["cluster_id"]: s for s in read_xlsx(args.carry) if s.get("decision", "").upper() in ("K", "D")}
                for r in rows:
                    o = old.get(r["cluster_id"])
                    if o:
                        r["decision"], r["rule"], r["note"] = o["decision"].upper(), o.get("rule", "").upper(), o.get("note", "")
                        carried += 1
            target = extended_targets(cell_stats(rows), quota, args.margin)  # with the top-up for short years, from the carried decisions
            for r in rows:
                r["target_K"] = target[r["cell"]]
            columns = QUOTA_COLUMNS
            no_quota = sorted({r["cell"] for r in rows if r["cell"] not in quota})
            extra = (f"; quota mode: random order (seed {args.order_seed}), target = queries + {args.margin}, at most the cell's rows"
                     + (f"; {carried} decisions carried from {args.carry}" if args.carry else "")
                     + (f"; cells with no queries in the allocation: {', '.join(no_quota)}" if no_quota else ""))
        write_xlsx(args.out, rows, columns)
        low = sum(1 for r in rows if r["words_in_passage"] < LOW_MATCH)
        some = sum(1 for r in rows if LOW_MATCH <= r["words_in_passage"] < SOME_MATCH)
        print(f"[review] sheet with {len(rows)} rows; statement words in the passage: {low} under {LOW_MATCH}% (orange), "
              f"{some} from {LOW_MATCH} to {SOME_MATCH - 1}% (yellow){extra}: {args.out}")
        return 0

    if args.command == "import":
        merged = {r["cluster_id"]: r for r in load_merged(args.work)}
        sheet = read_xlsx(args.sheet)
        quota_mode = bool(sheet) and "order" in sheet[0]
        if quota_mode and not args.allocation:
            print("[review] this sheet is in quota mode (it has an order column): give --allocation as well")
            return 1
        reviewed, undecided, errors = [], [], []
        for s in sheet:
            cid = s.get("cluster_id", "")
            if cid not in merged:
                errors.append(f"{cid}: unknown cluster")
                continue
            dec = s.get("decision", "").upper()
            rule = s.get("rule", "").upper()
            if quota_mode and dec == "":
                if rule or s.get("note", ""):
                    errors.append(f"{cid}: a rule or a note without a decision")
                undecided.append(cid)
                continue
            if dec not in ("K", "D"):
                errors.append(f"{cid}: decision must be K or D (got '{dec}')")
                continue
            if dec == "D" and rule not in RULES:
                errors.append(f"{cid}: a drop needs a rule D1-D5 (got '{rule}')")
                continue
            reviewed.append({**merged[cid], "decision": dec, "rule": rule if dec == "D" else "", "note": s.get("note", ""),
                             "reviewer": args.reviewer, "reviewed_on": dt.date.today().isoformat()})
        on_sheet = {r["cluster_id"] for r in reviewed} | set(undecided)
        missing = set(merged) - {s.get("cluster_id", "") for s in sheet} if quota_mode else set(merged) - on_sheet
        status: dict[str, dict] = {}
        outside: list[str] = []
        if quota_mode:
            # the order must be the seeded one (a sort or an edit in Excel would break the random order the rule rests on)
            expected = {r["cluster_id"]: r["order"] for r in random_order([dict(m) for m in merged.values()], args.order_seed)}
            for s in sheet:
                if s.get("cluster_id") in expected and str(expected[s["cluster_id"]]) != s.get("order", "").split(".")[0]:
                    errors.append(f"{s['cluster_id']}: order {s.get('order')} is not the seeded order {expected[s['cluster_id']]} "
                                  f"(was the column edited, or a different --order-seed used?)")
            quota = quotas(args.allocation)
            decided = {r["cluster_id"]: r for r in reviewed}
            by_cell_rows: dict[str, list[str]] = {}
            for cid in sorted(on_sheet & set(merged), key=lambda c: expected.get(c, 0)):
                by_cell_rows.setdefault(merged[cid]["cell"], []).append(cid)
            stats = {}
            for cell, cids in sorted(by_cell_rows.items()):
                prefix = 0
                while prefix < len(cids) and cids[prefix] in decided:
                    prefix += 1
                for i, cid in enumerate(cids):
                    if cid in decided:
                        decided[cid]["eligible"] = i < prefix
                        if i >= prefix:
                            outside.append(cid)
                k = sum(1 for cid in cids[:prefix] if decided[cid]["decision"] == "K")
                stats[cell] = {"rows": len(cids), "reviewed_from_top": prefix, "K_from_top": k}
            targets = extended_targets(stats, quota, args.margin)
            for cell, s in stats.items():
                k, prefix, target = s["K_from_top"], s["reviewed_from_top"], targets[cell]
                state = "target reached" if k >= target else ("all rows reviewed" if prefix == s["rows"] else "continue")
                status[cell] = {**s, "queries": quota.get(cell, 0), "target_K": target, "status": state}
            reviewed += [{**merged[cid], "decision": "", "rule": "", "note": "", "eligible": False} for cid in undecided if cid in merged]
        errors += d3_twin_errors(reviewed, quota_mode)
        if errors or missing:
            for e in errors[:30]:
                print("  " + e)
            print(f"[review] {len(errors)} errors, {len(missing)} rows {'missing from the sheet' if quota_mode else 'without a valid decision'}; nothing written")
            return 1
        write_jsonl(d / "reviewed.jsonl", reviewed)
        done = [r for r in reviewed if r["decision"] in ("K", "D")]
        kept = sum(1 for r in done if r["decision"] == "K")
        by_rule: dict[str, int] = {}
        for r in done:
            if r["decision"] == "D":
                by_rule[r["rule"]] = by_rule.get(r["rule"], 0) + 1
        by_cell: dict[str, list[int]] = {}
        for r in done:
            t = by_cell.setdefault(r["cell"], [0, 0])
            t[0 if r["decision"] == "K" else 1] += 1
        lines = ["# Review of the drafted candidates", "", f"reviewer: {args.reviewer}; rows decided: {len(done)}; kept: {kept}; dropped: {len(done) - kept}", "",
                 "dropped by rule: " + ", ".join(f"{k} {v}" for k, v in sorted(by_rule.items())), "", "| cell | kept | dropped |", "|---|---|---|"]
        lines += [f"| {c} | {k} | {dd} |" for c, (k, dd) in sorted(by_cell.items())]
        if quota_mode:
            (d / "review_status.json").write_text(json.dumps(status, indent=1) + "\n", encoding="utf-8")
            todo = [c for c, v in status.items() if v["status"] == "continue"]
            lines += ["", f"## Quota mode (random order, seed {args.order_seed}; target = queries + {args.margin}, at most the cell's rows, plus the cell's share of its country's shortfall in closed years)", "",
                      f"rows not yet reviewed: {len(undecided)}; decided rows below an undecided one, logged but not used in the draw: {len(outside)}"
                      + (f" ({', '.join(outside[:20])})" if outside else ""), "",
                      "| cell | rows | reviewed from the top | K from the top | queries | target | status |", "|---|---|---|---|---|---|---|"]
            lines += [f"| {c} | {v['rows']} | {v['reviewed_from_top']} | {v['K_from_top']} | {v['queries']} | {v['target_K']} | {v['status']} |"
                      for c, v in sorted(status.items())]
            lines += ["", f"cells still to review: {len(todo)}" + (f" ({', '.join(todo)})" if todo else "")]
        (d / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[review] {len(done)} decisions imported, {kept} kept; {d / 'reviewed.jsonl'}")
        if quota_mode:
            print(f"[review] quota mode: {sum(1 for v in status.values() if v['status'] != 'continue')} of {len(status)} cells finished; "
                  f"still to review: {', '.join(todo) or 'none'}; {len(outside)} decided rows below a gap (not used); details in {d / 'review.md'}")
        return 0

    if args.command == "second":
        reviewed = [r for r in read_jsonl(d / "reviewed.jsonl") if r["decision"] in ("K", "D")]  # quota mode: only the rows decided
        rng = random.Random(args.seed)
        k = max(1, round(len(reviewed) * args.fraction))
        sample = rng.sample(reviewed, k)
        for r in sample:
            r["sources"] = "; ".join(r["docs"])
            r["cells"] = ", ".join(r.get("cells", [r["cell"]]))
            r["words_in_passage"] = words_in_passage(r["claim"], r["passage"])  # the same help as the first reviewer had
            r["decision"], r["rule"], r["note"] = "", "", ""
        write_xlsx(args.out, sample, COLUMNS)
        print(f"[review] blind sample of {k} rows (seed {args.seed}): {args.out}")
        return 0

    if args.command == "agreement":
        first = {r["cluster_id"]: r for r in read_jsonl(d / "reviewed.jsonl")}
        second = [s for s in read_xlsx(args.sheet) if s.get("cluster_id") in first and s.get("decision", "").upper() in ("K", "D")]
        a = [first[s["cluster_id"]]["decision"] for s in second]
        b = [s["decision"].upper() for s in second]
        kap = kappa(a, b)
        agree = sum(1 for x, y in zip(a, b) if x == y)
        text = (f"# Agreement between the two reviewers on the blind sample\n\nrows: {len(second)}; raw agreement: {agree}/{len(second)}; "
                f"Cohen's kappa (keep/drop): {kap:.3f}\n")
        (d / "review_agreement.md").write_text(text, encoding="utf-8")
        print(text)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
