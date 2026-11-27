"""
build_qrels.py -- turn the label files into the qrels files of the paper.

  qrels_ai.txt        the main judge's label for EVERY pair of the pool, all queries: the main tables. The same AI
                      configuration labels every system, and no human label is mixed in (Balog et al. 2025,
                      Section 5.1: no LLM labels used selectively to fill holes in human judgments).
  qrels_human.txt     the human label of every document of the check sample: the label of the query's owner (the
                      judge who reads the whole query), also for a shared document; no discussion (TREC's primary
                      assessor, Voorhees 2000). A separate table on the check queries.
  qrels_ai_check.txt  the main judge's label on exactly the documents of qrels_human.txt, for the side-by-side table.
  pool_depth.jsonl    one line per pooled pair: ranks in every system, in the check sample or not, the AI label, the
                      human label.
  qrels_summary.md    counts.

Labels 's' (skipped) and notes are ignored. If a judge labelled the same document twice (undo, or a 'repeat' check),
the LAST 'pool' answer counts (common/labels.py).

Usage:
  python qrels/build_qrels.py --work WORK --annotators R,M --llm-labels WORK/judge/labels_llm_<main judge>.jsonl
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonl import read_jsonl, write_jsonl  # noqa: E402
from common.trec import write_qrels  # noqa: E402
from common.labels import final_check_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--annotators", default="R,M")
    ap.add_argument("--llm-labels", required=True, help="the main judge's labels (judge/validate_judge.py names it)")
    ap.add_argument("--out", default=None, help="default WORK/qrels")
    args = ap.parse_args()
    work = Path(args.work)
    out = Path(args.out) if args.out else work / "qrels"
    annotators = [x.strip() for x in args.annotators.split(",") if x.strip()]
    pool = list(read_jsonl(work / "pool" / "pool.jsonl"))
    llm = {(r["query_id"], r["doc_id"]): r["label"] for r in read_jsonl(args.llm_labels) if r.get("label") in ("y", "n")}
    final, undecided = final_check_labels(work, annotators)
    if undecided:
        print(f"note: {len(undecided)} shared documents were labelled only by the second judge (the owner stopped earlier): they have no human label")

    qa: dict[str, dict[str, int]] = {}
    qh: dict[str, dict[str, int]] = {}
    qac: dict[str, dict[str, int]] = {}
    rows, c = [], Counter()
    for r in pool:
        k = (r["query_id"], r["doc_id"])
        a_lab, h_lab = llm.get(k), final.get(k)
        if a_lab:
            qa.setdefault(k[0], {})[k[1]] = 1 if a_lab == "y" else 0
        else:
            c["pool pairs without an AI label"] += 1
        if h_lab:
            qh.setdefault(k[0], {})[k[1]] = 1 if h_lab == "y" else 0
            if a_lab:
                qac.setdefault(k[0], {})[k[1]] = 1 if a_lab == "y" else 0
        rows.append({"query_id": k[0], "doc_id": k[1], "ranks": r["ranks"], "min_rank": r["min_rank"], "check": r.get("check", False),
                     "ai_label": a_lab, "human_label": h_lab})
    write_qrels(out / "qrels_ai.txt", qa)
    write_qrels(out / "qrels_human.txt", qh)
    write_qrels(out / "qrels_ai_check.txt", qac)
    write_jsonl(out / "pool_depth.jsonl", rows)
    rel = lambda q: sum(v for d in q.values() for v in d.values())  # noqa: E731
    lines = ["# Qrels summary", "",
             f"AI labels (main judge, {Path(args.llm_labels).name}): {sum(len(v) for v in qa.values())} pairs of {len(pool)} "
             f"({rel(qa)} relevant) over {len(qa)} queries; pairs without an AI label: {c['pool pairs without an AI label']}",
             f"human labels (check sample): {sum(len(v) for v in qh.values())} documents ({rel(qh)} relevant) over {len(qh)} queries; "
             f"the AI's labels on the same documents: {rel(qac)} relevant",
             f"queries with at least one relevant document: AI {sum(1 for v in qa.values() if any(v.values()))} of {len({r['query_id'] for r in pool})}, "
             f"human {sum(1 for v in qh.values() if any(v.values()))} of {len(qh)}"]
    (out / "qrels_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:]))
    print(f"wrote {out / 'qrels_ai.txt'}, qrels_human.txt, qrels_ai_check.txt, pool_depth.jsonl")


if __name__ == "__main__":
    main()
