"""
adjudicate.py -- agreement between the human judges on the documents they both judged.

IN THE PAPER there is no discussion. The final human label of every checked document is the label
of the query's owner, the judge who reads the whole query, as TREC's primary assessor (common/labels.final_check_labels);
the second judge's labels on the shared documents only measure the agreement. What the paper uses from this script is
the agreement report (agreement.md, agreement.json). The discussion below (to_adjudicate.jsonl, --discussion-sheet,
--decisions) is kept as an option for a study that wants agreed labels; its labels_adjudicated.jsonl is not read by
the qrels or the validation.

Reads WORK/labels/labels_<annotator>.jsonl for the judges given, keeps
the last 'pool' answer of each judge per document, and for the shared
documents (WORK/pool/shared.json, written by pool/build_pool.py: the 150
documents of the check sample judged by both):
  - the agreed label when the judges agree (with three or more judges: a
    strict majority);
  - otherwise the document goes to WORK/labels/to_adjudicate.jsonl; the two
    judges discuss it and agree one label, written back through --decisions
    (a JSONL with query_id, doc_id, label). The agreement numbers are always
    computed on the labels given independently, before the discussion.
Also reports: pairwise Cohen's kappa between the judges (its 95 percent
bootstrap interval, which the AI judge's pass test uses, is computed by
judge/validate_judge.py), each judge's accuracy on gold items and
consistency on repeated items, and seconds per item.

--exclusions (not used in the paper, whose two human judges wrote no formula; kept for
a judge who also wrote formulas: {annotator: [query_id, ...]}, the queries whose
formula that annotator wrote): on a shared document, the labels of excluded
annotators count for the kappa but NEVER for the final label; the final
label is the majority of the other annotators, or the single other label
when only one remains (source "non-writer"). The kappa of every pair is also
given split by who wrote the formula (a wrote / b wrote / neither), so a
writer's bias on their own queries would be visible.

THE DISCUSSION SHEET (optional, --discussion-sheet): when documents are still to discuss, the script also writes
WORK/labels/to_discuss.xlsx, one row per document: the query (with --queries), the claim, the start of the
explanation, its date and fact-checker, each judge's vote, an empty column 'agreed' and the ids. The judges discuss
each row, type y or n in 'agreed', SAVE THE SHEET AS decisions.xlsx in the same folder, and run the script again
with --decisions WORK/labels/decisions.xlsx (a JSONL with query_id, doc_id, label is accepted too). The sheet is
never written over the file given to --decisions.

Output: WORK/labels/labels_adjudicated.jsonl, WORK/labels/agreement.md, agreement.json, to_adjudicate.jsonl,
to_discuss.xlsx (only with --discussion-sheet, when something is still to discuss)

Usage: python judge/adjudicate.py --work WORK --annotators R,M          (the paper: the agreement report only)
       python judge/adjudicate.py --work WORK --annotators R,M --discussion-sheet [--queries Q.json] [--decisions WORK/labels/decisions.xlsx]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonl import read_jsonl, write_jsonl, write_json, read_json  # noqa: E402
from judge.agreement import cohen_kappa  # noqa: E402
from common.labels import load_annotator  # noqa: E402


def write_discussion_sheet(path: Path, rows: list[dict], annotators: list[str], labels: dict, work: Path, queries: str | None) -> None:
    """One row per document the judges disagree on, with what they need to discuss it and an empty 'agreed' column."""
    import openpyxl
    from openpyxl.styles import Alignment, Font
    pool = {(r["query_id"], r["doc_id"]): r for r in read_jsonl(work / "pool" / "pool.jsonl")}
    query_text = {}
    if queries:
        from common.queries import load_queries
        query_text = {q["query_id"]: q.get("free_text") for q in load_queries(queries)}
    cols = ["query_id", "query", "claim", "explanation (start)", "date", "fact-checker"] + [f"vote {a}" for a in annotators] + ["agreed", "doc_id"]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "to discuss"
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in rows:
        k = (r["query_id"], r["doc_id"])
        d = pool.get(k, {})
        ws.append([r["query_id"], query_text.get(r["query_id"], ""), d.get("claim_text", ""), d.get("article_start", ""),
                   d.get("date_published", ""), d.get("author_name", "")] + [labels[a].get(k, "") for a in annotators] + ["", r["doc_id"]])
    for letter, width in zip("ABCDEFGHIJKL", [16, 45, 50, 60, 11, 18] + [8] * len(annotators) + [8, 30]):
        ws.column_dimensions[letter].width = width
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def read_sheet_decisions(path: Path) -> dict:
    """(query_id, doc_id) -> 'y' or 'n' from the filled discussion sheet; rows with an empty or other 'agreed' are skipped."""
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0]
    it = ws.iter_rows(values_only=True)
    head = [str(h or "").strip() for h in next(it)]
    iq, idoc, iag = head.index("query_id"), head.index("doc_id"), head.index("agreed")
    out = {}
    for row in it:
        if not row or row[iq] is None:
            continue
        lab = str(row[iag] or "").strip().lower()
        if lab in ("y", "n"):
            out[(str(row[iq]).strip(), str(row[idoc]).strip())] = lab
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--annotators", default="A,B,C")
    ap.add_argument("--decisions", default=None, help="the agreed labels: the filled discussion sheet (.xlsx, column 'agreed') or a JSONL")
    ap.add_argument("--queries", default=None, help="the query file, to show each query in the discussion sheet")
    ap.add_argument("--discussion-sheet", action="store_true", help="write to_discuss.xlsx for a discussion (not used in the paper)")
    ap.add_argument("--exclusions", default=None, help="pool exclusions file: an annotator's own-formula labels count for kappa only")
    args = ap.parse_args()
    work = Path(args.work)
    exclusions = {a: set(v) for a, v in (read_json(args.exclusions) if args.exclusions else {}).items()}
    annotators = [a.strip() for a in args.annotators.split(",") if a.strip()]
    overlap = [tuple(x) for x in read_json(work / "pool" / "shared.json")]
    labels: dict[str, dict[tuple, str]] = {}
    gold_hits, repeat_pairs, seconds = defaultdict(list), defaultdict(list), defaultdict(list)
    for a in annotators:
        p = work / "labels" / f"labels_{a}.jsonl"
        labels[a] = {}
        if not p.exists():
            print(f"warning: no label file for {a}")
            continue
        la = load_annotator(p)
        labels[a] = la["pool"]
        seconds[a] = la["seconds"]
        gold_hits[a] = [lab == gold for lab, gold in la["gold"]]
        repeat_pairs[a] = [(labels[a][(q, d)], lab) for q, d, lab in la["repeat"] if (q, d) in labels[a]]
    decisions = {}
    if args.decisions and not Path(args.decisions).exists():
        raise SystemExit(f"--decisions: {args.decisions} does not exist")
    if args.decisions and args.decisions.lower().endswith(".xlsx"):
        decisions = read_sheet_decisions(Path(args.decisions))
    elif args.decisions:
        decisions = {(r["query_id"], r["doc_id"]): r["label"] for r in read_jsonl(args.decisions) if r.get("label") in ("y", "n")}
    final, to_adjudicate = [], []
    for key in overlap:
        votes = [labels[a][key] for a in annotators if key in labels[a]]
        eligible = [labels[a][key] for a in annotators if key in labels[a] and key[0] not in exclusions.get(a, set())]
        c = Counter(eligible)
        if key in decisions:
            final.append({"query_id": key[0], "doc_id": key[1], "label": decisions[key], "source": "decision", "votes": votes})
        elif eligible and c.most_common(1)[0][1] * 2 > len(eligible) and len(eligible) >= 2:
            final.append({"query_id": key[0], "doc_id": key[1], "label": c.most_common(1)[0][0], "source": "majority", "votes": votes})
        elif len(eligible) == 1 and len(votes) > 1:
            final.append({"query_id": key[0], "doc_id": key[1], "label": eligible[0], "source": "non-writer", "votes": votes})
        else:
            to_adjudicate.append({"query_id": key[0], "doc_id": key[1], "votes": votes, "label": None})
    write_jsonl(work / "labels" / "labels_adjudicated.jsonl", final)
    write_jsonl(work / "labels" / "to_adjudicate.jsonl", to_adjudicate)
    sheet = work / "labels" / "to_discuss.xlsx"
    if args.discussion_sheet and to_adjudicate and not (args.decisions and Path(args.decisions).resolve() == sheet.resolve()):
        write_discussion_sheet(sheet, to_adjudicate, annotators, labels, work, args.queries)
        print(f"{len(to_adjudicate)} shared documents to discuss: open {sheet}, type y or n in 'agreed', "
              f"save it as decisions.xlsx in the same folder and run again with --decisions {sheet.parent / 'decisions.xlsx'}")
    rep = {"annotators": annotators, "overlap": len(overlap), "adjudicated": len(final), "to_adjudicate": len(to_adjudicate), "pairwise": {}}
    kappas = []
    rep["pairwise_by_writer"] = {}
    for a, b in combinations(annotators, 2):
        both = [k for k in overlap if k in labels[a] and k in labels[b]]
        pairs = [(labels[a][k], labels[b][k]) for k in both]
        rep["pairwise"][f"{a}-{b}"] = cohen_kappa(pairs)
        if rep["pairwise"][f"{a}-{b}"]["kappa"] is not None:
            kappas.append(rep["pairwise"][f"{a}-{b}"]["kappa"])
        if exclusions:
            split = {f"{a} wrote the formula": [k for k in both if k[0] in exclusions.get(a, set())],
                     f"{b} wrote the formula": [k for k in both if k[0] in exclusions.get(b, set())]}
            split["neither wrote it"] = [k for k in both if k not in set(split[f"{a} wrote the formula"]) | set(split[f"{b} wrote the formula"])]
            rep["pairwise_by_writer"][f"{a}-{b}"] = {name: cohen_kappa([(labels[a][k], labels[b][k]) for k in ks]) for name, ks in split.items()}
    rep["human_human_kappa_mean"] = round(statistics.mean(kappas), 4) if kappas else None
    rep["per_annotator"] = {a: {"items": len(labels[a]), "positive_rate": round(sum(1 for v in labels[a].values() if v == "y") / len(labels[a]), 4) if labels[a] else None,
                                "gold_accuracy": round(sum(gold_hits[a]) / len(gold_hits[a]), 4) if gold_hits[a] else None, "gold_items": len(gold_hits[a]),
                                "repeat_consistency": round(sum(1 for x, y in repeat_pairs[a] if x == y) / len(repeat_pairs[a]), 4) if repeat_pairs[a] else None,
                                "median_seconds": statistics.median(seconds[a]) if seconds[a] else None} for a in annotators}
    write_json(work / "labels" / "agreement.json", rep)
    lines = ["# Annotator agreement", "", f"shared documents: {len(overlap)}; with an agreed label (the judges agree, or a discussion "
             f"decided it with --decisions): {len(final)}; without one: {len(to_adjudicate)}. Without a discussion the final human "
             f"label of a checked document is the label of the query's owner (common/labels.final_check_labels).", "",
             "| pair | n | kappa | observed agreement |", "|---|---:|---:|---:|"]
    for k, v in rep["pairwise"].items():
        lines.append(f"| {k} | {v['n']} | {v['kappa']} | {v['observed']} |")
    for pair, d in rep["pairwise_by_writer"].items():
        for name, v in d.items():
            lines.append(f"| {pair}, {name} | {v['n']} | {v['kappa']} | {v['observed']} |")
    lines += ["", f"mean human-human kappa: {rep['human_human_kappa_mean']}", "", "| annotator | items | positive rate | gold accuracy (n) | repeat consistency | median seconds |", "|---|---:|---:|---:|---:|---:|"]
    for a, v in rep["per_annotator"].items():
        lines.append(f"| {a} | {v['items']} | {v['positive_rate']} | {v['gold_accuracy']} ({v['gold_items']}) | {v['repeat_consistency']} | {v['median_seconds']} |")
    (work / "labels" / "agreement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
