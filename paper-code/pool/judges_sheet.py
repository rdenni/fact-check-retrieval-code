"""
judges_sheet.py -- the human judges label in an Excel sheet (no Python and no terminal needed, so a judge who is not
a programmer can label).

  make   per judge, from WORK/pool (pool.jsonl, assignments.json, check.json), two files:
         HOW-TO-JUDGE_<id>.txt, the instructions in plain words and the whole guideline (pool/GUIDELINE.md);
         sheet_<id>.xlsx, ONE tab, "judge" (it opens directly; a second tab is easy to miss),
         with one row per item: the queries in
         the order of the check draw, all the items of one query in a row, inside a query a random order
         fixed by the judge's id, and --repeat items shown a second time at the boundary before a later query
         in the second half (to measure the judge's own consistency). Each row shows what the AI judge reads
         (judge/prompts.py): the query, its background in a column of its own (empty for most queries), the
         claim, the first 500 characters of the explanation (the fact-checker's text), the organisation the fact-checker works
         for, its country and the date. Never the system, a rank, a formula or the verdict.
         The judge types y or n in the column "answer" (a drop-down accepts nothing else) and may write in "note".
         Every other cell is locked (sheet protection without a password), so a text cannot be changed by mistake.
         The texts are wrapped and the rows are tall enough to read them whole: no cell has to be opened.
  read   reads a returned sheet into WORK/labels/labels_<id>.jsonl, in the format the tool writes (common/labels.py
         reads both alike): one line per answered row, kind 'pool' or 'repeat', the note, the guideline's sha256;
         'seconds' is empty, because a sheet cannot time an answer (only the writers' time is reported). Answers are read case-blind; 'yes'/'no' count as y/n; anything else is listed and
         skipped; empty rows are counted (stopping early is allowed: after query 15, 20 or 25 of the check).
         An existing label file is never overwritten: it is renamed with a time stamp first.
  A SECOND ROUND (make --only R --round 2): a fresh blank sheet for one judge, with the same items, another random
         order of the rows inside each query and other repeated items (the order is seeded with the round), written
         as sheet_R_round2.xlsx; read takes it like any sheet of that judge, and each label records the sheet's
         file name. Without --round the order is the first round's, so the first sheets can be made again.

Usage:
  python pool/judges_sheet.py make --work WORK --queries Q.json --out FOLDER [--repeat 10] [--only R] [--round 2]
  python pool/judges_sheet.py read --work WORK --sheet FOLDER/sheet_M.xlsx --annotator M [--out FILE]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import math
import re
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonl import read_json, read_jsonl, write_jsonl  # noqa: E402
from common.queries import load_queries  # noqa: E402

HERE = Path(__file__).resolve().parent
GUIDELINE = HERE / "GUIDELINE.md"
# Column names, in the words of the guideline (pool/GUIDELINE.md); the position in the check draw is "query n." (the
# number the stopping rule uses). "answer" sits right after "claim", the text the judge decides on, so the eye does not
# travel across the sheet (2026-11-20). The three id columns at the end are hidden.
COLS = [("row", 6), ("query n.", 9), ("query", 34), ("background", 24), ("claim", 40), ("answer", 9),
        ("explanation (start)", 55), ("organisation", 14), ("country", 12), ("date", 11), ("note", 24),
        ("query_id", 1), ("doc_id", 1), ("kind", 1)]
NAMES = [c for c, _ in COLS]
ANSWER_COL, NOTE_COL = NAMES.index("answer") + 1, NAMES.index("note") + 1
WIDTH = {name: w for name, w in COLS}
FIRST_ROW = 2  # row 1 is the header


def guideline_sha256() -> str:
    """sha256 of pool/GUIDELINE.md (line ends made equal), recorded with every answer, so that a changed guideline is
    visible; the AI prompt (judge/prompts.py) has the same rules."""
    return hashlib.sha256(GUIDELINE.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest()


def item_order(annotator: str, items: list[tuple], drawn: list[str], n_repeat: int,
               judging_round: int = 1) -> list[tuple[str, tuple]]:
    """The order of the rows: the queries in the order of the check draw, all the items of one query together, inside a
    query a random order fixed by the judge's id (and, from the second round on, by the round); the repeated items go
    between two queries, in the second half."""
    rng = random.Random(f"{annotator}-{len(items)}" if judging_round == 1 else f"{annotator}-round{judging_round}-{len(items)}")
    by_query: dict[str, list] = {}
    for it in items:
        by_query.setdefault(it[0], []).append(it)
    query_order = [q for q in drawn if q in by_query] + sorted(q for q in by_query if q not in drawn)
    groups = []
    for qid in query_order:
        its = list(by_query[qid])
        rng.shuffle(its)
        groups.append([("pool", it) for it in its])
    for it in rng.sample(items, min(n_repeat, len(items))):
        groups.insert(rng.randrange(len(groups) // 2, len(groups) + 1), [("repeat", it)])
    return [x for g in groups for x in g]


def lines_needed(text: str, width: int) -> int:
    """How many wrapped lines a text takes in a column of this width. Generous on purpose (0.85 characters per width
    unit, one spare line): Excel fits about 1.1 with Calibri, but another default font or
    zoom fits fewer, and a cut text forces the judge to open the cell."""
    per_line = max(10, int(width * 0.85))
    return 1 + sum(max(1, math.ceil(len(part) / per_line)) for part in (text or "").split("\n"))


README = """HOW TO JUDGE  (judge {annotator}, file {sheet})

What you do
- Open the Excel file. Each row shows a query, a claim that a fact-checker checked, and the other columns
  that the guideline below describes.
- For each row, decide with the guideline, then type  y  (yes) or  n  (no) in the column "answer" and press
  Enter. Excel moves to the next row by itself.
- The column "note" is optional. Write "unclear" there when you cannot understand the claim (rule 4 of the
  guideline), and anything else you want to tell us.
- Read the guideline once before you start. It is short.

How the sheet is arranged
- The rows of one query come together. Where a new query starts, a line appears above the row and the
  colour of the rows changes (white or light blue).
- The column "query n." says which query you are on, for example "3 of 25".
- The column "row" counts the rows. There are {n_rows}.
- About {n_repeat} rows appear a second time, on purpose. Answer them as if they were new. Do not scroll back to
  your earlier answer.

Practical rules
- Do not use an AI assistant. Do not search the web.
- You can save at any time (Ctrl+S) and continue later. Do not sort, move or delete rows.
- If you must stop before the end, stop at the end of a query, after query 15, 20 or 25, and agree with the
  other judge where you both stop.
- When you have finished, save the file and send it back as it is.

"""


def answered_rows(path: Path) -> int:
    """How many rows of an existing sheet already hold an answer (0 when the file does not exist)."""
    import openpyxl
    if not path.exists():
        return 0
    ws = openpyxl.load_workbook(path, read_only=True)["judge"]
    rows = ws.iter_rows(values_only=True)
    head = [str(h or "") for h in next(rows)]
    i = head.index("answer")
    return sum(1 for r in rows if r and r[i] not in (None, ""))


def make(args):
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Protection, Side
    from openpyxl.worksheet.datavalidation import DataValidation
    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # never write over a judge's work: a sheet that already holds answers stops the script
    assignments_ids = list(read_json(work / "pool" / "assignments.json"))
    if args.only:
        if args.only not in assignments_ids:
            raise SystemExit(f"--only {args.only}: no such judge in assignments.json ({', '.join(assignments_ids)})")
        assignments_ids = [args.only]
    if args.round < 1:
        raise SystemExit("--round starts at 1")
    suffix = "" if args.round == 1 else f"_round{args.round}"
    started = {a: answered_rows(out / f"sheet_{a}{suffix}.xlsx") for a in assignments_ids}
    if any(started.values()):
        raise SystemExit(f"these sheets already hold answers, so they are not written over: "
                         f"{', '.join(f'sheet_{a}{suffix}.xlsx ({n} answers)' for a, n in started.items() if n)}; choose another --out folder")
    queries = {q["query_id"]: q for q in load_queries(args.queries)}
    pool = {(r["query_id"], r["doc_id"]): r for r in read_jsonl(work / "pool" / "pool.jsonl")}
    assignments = read_json(work / "pool" / "assignments.json")
    drawn = read_json(work / "pool" / "check.json")["queries"]
    position = {q: i + 1 for i, q in enumerate(drawn)}
    guideline = GUIDELINE.read_text(encoding="utf-8")
    fills = [PatternFill("solid", fgColor="FFFFFF"), PatternFill("solid", fgColor="EEF3F8")]
    answer_fill = PatternFill("solid", fgColor="FFF7D6")
    top = Border(top=Side(style="medium", color="5B7FA6"))
    wrap = Alignment(wrap_text=True, vertical="top")
    for annotator, pairs in assignments.items():
        if annotator not in assignments_ids:
            continue
        items = [tuple(x) for x in pairs]
        missing = [it for it in items if it not in pool]
        if missing:
            raise SystemExit(f"{len(missing)} items of {annotator} are not in pool.jsonl: rebuild the pool")
        order = item_order(annotator, items, drawn, args.repeat, args.round)
        n_repeat = sum(1 for k, _ in order if k == "repeat")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "judge"
        ws.append([c for c, _ in COLS])
        for i, (name, width) in enumerate(COLS, 1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = width
            ws.cell(1, i).font = Font(bold=True)
            ws.cell(1, i).fill = PatternFill("solid", fgColor="D9E2EC")
        for name in ("query_id", "doc_id", "kind"):
            ws.column_dimensions[openpyxl.utils.get_column_letter(NAMES.index(name) + 1)].hidden = True
        group, last_q = -1, None
        for n, (kind, (qid, doc)) in enumerate(order, 1):
            q, d = queries.get(qid, {}), pool[(qid, doc)]
            query_cell = q.get("free_text", "")
            background = q.get("background") or ""
            claim = d.get("claim_text") or "(empty)"
            explanation = (d.get("article_start") or "(no explanation)").replace("\n", " ")
            r = FIRST_ROW + n - 1
            values = {"row": n, "query n.": f"{position.get(qid, '?')} of {len(drawn)}", "query": query_cell,
                      "background": background, "claim": claim, "answer": None, "explanation (start)": explanation,
                      "organisation": d.get("author_name") or "?", "country": d.get("country") or "?",
                      "date": d.get("date_published") or "?", "note": None, "query_id": qid, "doc_id": doc, "kind": kind}
            ws.append([values[name] for name in NAMES])  # by name, so the column order is set in COLS only
            new_group = qid != last_q
            if new_group:
                group += 1
                last_q = qid
            for c in range(1, len(COLS) + 1):
                cell = ws.cell(r, c)
                # the hidden id columns are not wrapped: wrapped in a 1-wide column, they would make Excel's
                # "fit row height" blow every row up
                cell.alignment = wrap if c <= NOTE_COL else Alignment(vertical="top")
                cell.fill = fills[group % 2]
                if new_group:
                    cell.border = top
            for c in (ANSWER_COL, NOTE_COL):  # the only cells the judge can change
                ws.cell(r, c).protection = Protection(locked=False)
            ws.cell(r, ANSWER_COL).fill = answer_fill
            ws.cell(r, ANSWER_COL).alignment = Alignment(horizontal="center", vertical="top")
            ws.row_dimensions[r].height = min(409, 15 * max(lines_needed(query_cell, WIDTH["query"]),
                                                            lines_needed(background, WIDTH["background"]),
                                                            lines_needed(claim, WIDTH["claim"]),
                                                            lines_needed(explanation, WIDTH["explanation (start)"])))
        last = FIRST_ROW + len(order) - 1
        dv = DataValidation(type="list", formula1='"y,n"', allow_blank=True, showErrorMessage=True,
                            errorTitle="y or n", error="Type y (yes) or n (no).")
        ws.add_data_validation(dv)
        col = openpyxl.utils.get_column_letter(ANSWER_COL)
        dv.add(f"{col}{FIRST_ROW}:{col}{last}")
        ws.freeze_panes = "C2"
        ws.protection.sheet = True
        ws.protection.formatColumns = False  # the judge may widen a column or a row to read better
        ws.protection.formatRows = False
        # the guideline the sheet was made with, so that read can refuse a sheet judged under another guideline
        wb.properties.keywords = f"guideline sha256 {guideline_sha256()}"
        path = out / f"sheet_{annotator}{suffix}.xlsx"
        wb.save(path)
        howto = README.format(annotator=annotator, sheet=path.name, n_repeat=n_repeat, n_rows=len(order))
        howto += "GUIDELINE (the same one the AI judge follows)\n=============================================\n\n" + re.sub(r"(?m)^#+ ", "", guideline.replace("**", ""))
        (out / f"HOW-TO-JUDGE_{annotator}{suffix}.txt").write_text(howto, encoding="utf-8")
        print(f"{path} and HOW-TO-JUDGE_{annotator}{suffix}.txt: {len(order)} rows ({len(items)} items + {n_repeat} repeated) over {len({q for q, _ in items})} queries")


def read(args):
    import openpyxl
    work = Path(args.work)
    assigned = {tuple(x) for x in read_json(work / "pool" / "assignments.json").get(args.annotator, [])}
    if not assigned:
        raise SystemExit(f"annotator {args.annotator!r} has no items in assignments.json")
    wb = openpyxl.load_workbook(args.sheet, read_only=True, data_only=True)
    fingerprint = guideline_sha256()
    made_with = str(wb.properties.keywords or "").replace("guideline sha256 ", "").strip()
    if made_with and made_with != fingerprint:
        raise SystemExit(f"the sheet was made with the guideline sha256 {made_with[:12]}, but pool/GUIDELINE.md now has "
                         f"{fingerprint[:12]}: the answers were given under another guideline. Restore the guideline the "
                         "sheet was made with (git) before reading it.")
    if not made_with:
        print("note: this sheet records no guideline fingerprint (made before 2026-11-19); the current one is recorded")
    ws = wb["judge"]
    rows = ws.iter_rows(values_only=True)
    head = [str(h or "") for h in next(rows)]
    ix = {name: head.index(name) for name in ("row", "query n.", "answer", "note", "query_id", "doc_id", "kind")}
    ts = dt.datetime.fromtimestamp(Path(args.sheet).stat().st_mtime, dt.timezone.utc).isoformat(timespec="seconds")
    out_rows, empty, bad, foreign, seen = [], [], [], [], set()
    for row in rows:
        if row is None or row[ix["query_id"]] is None:
            continue
        qid, doc, kind = str(row[ix["query_id"]]), str(row[ix["doc_id"]]), str(row[ix["kind"]] or "pool")
        if (qid, doc) not in assigned:
            foreign.append(row[ix["row"]])
            continue
        raw = str(row[ix["answer"]] or "").strip().lower()
        note = str(row[ix["note"]] or "").strip() or None
        label = {"y": "y", "yes": "y", "n": "n", "no": "n"}.get(raw)
        if raw and label is None:
            bad.append((row[ix["row"]], raw))
        if label is None:
            if not raw:
                empty.append(row[ix["query n."]])
            if note:
                out_rows.append({"annotator": args.annotator, "query_id": qid, "doc_id": doc, "label": "note", "seconds": None,
                                 "ts": ts, "kind": kind, "note": note, "guideline_sha256": fingerprint, "source": "sheet",
                                 "sheet": Path(args.sheet).name})
            continue
        if kind == "pool":
            seen.add((qid, doc))
        out_rows.append({"annotator": args.annotator, "query_id": qid, "doc_id": doc, "label": label, "seconds": None,
                         "ts": ts, "kind": kind, "note": note, "guideline_sha256": fingerprint, "source": "sheet",
                         "sheet": Path(args.sheet).name})
    if foreign:
        raise SystemExit(f"rows {foreign[:5]} are not items of {args.annotator}: is this {args.annotator}'s sheet?")
    out = Path(args.out) if args.out else work / "labels" / f"labels_{args.annotator}.jsonl"
    if out.exists():
        keep = out.with_name(out.stem + dt.datetime.now().strftime("_before_%Y%m%d-%H%M%S") + out.suffix)
        out.rename(keep)
        print(f"the existing {out.name} was kept as {keep.name}")
    write_jsonl(out, out_rows)
    n_lab = sum(1 for r in out_rows if r["label"] in ("y", "n"))
    print(f"{args.annotator}: {n_lab} answers read ({len(seen)} of {len(assigned)} items, "
          f"{sum(1 for r in out_rows if r['kind'] == 'repeat' and r['label'] in ('y', 'n'))} repeated), "
          f"{sum(1 for r in out_rows if r['note'])} notes -> {out}")
    if empty:
        stops = sorted({e for e in empty}, key=lambda s: int(str(s).split()[0]) if str(s).split()[0].isdigit() else 999)
        print(f"rows without an answer: {len(empty)}, in the queries {', '.join(map(str, stops[:10]))}")
    if bad:
        print(f"WARNING: answers that are not y or n were skipped: {bad[:10]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--work", required=True)
    m.add_argument("--queries", required=True)
    m.add_argument("--out", required=True, help="folder for sheet_<id>.xlsx")
    m.add_argument("--repeat", type=int, default=10, help="items shown a second time, to measure each judge's own consistency")
    m.add_argument("--only", default=None, help="make the sheet of this judge only (e.g. R)")
    m.add_argument("--round", type=int, default=1, help="2 or more: a fresh sheet with another row order, named sheet_<id>_round<N>.xlsx")
    r = sub.add_parser("read")
    r.add_argument("--work", required=True)
    r.add_argument("--sheet", required=True)
    r.add_argument("--annotator", required=True)
    r.add_argument("--out", default=None, help="default WORK/labels/labels_<annotator>.jsonl")
    args = ap.parse_args()
    make(args) if args.cmd == "make" else read(args)


if __name__ == "__main__":
    main()
