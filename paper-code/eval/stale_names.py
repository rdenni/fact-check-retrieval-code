"""
stale_names.py -- what the archive's old entity names cost the graph.

The archive's entities keep the names that some pages had earlier ("Twitter",
"Autism spectrum"); some pages have been renamed since ("X (social network)", "Autism"). The import
(queries/import_formulas.py --resolve-redirects) uses every name of a page, so the main run finds a fact-check
linked under any of them. This script compares that main run with the same formulas run with TODAY'S PAGES ONLY
(systems/graph_boolean.py --current-names-only), which is what the graph would give if its entities were
current and nothing translated the old names. The difference is the cost of the stale names, a defect of the
resource, reported as one extra line; the main tables use the main run.

Only the AI judge's labels are used (--qrels WORK/qrels/qrels_ai.txt). No new judging is
needed: a formula without some names can only return fewer documents (the formulas have no NOT; with a NOT the
script checks the subset property and says so), and every document of the main run's sets is in the pool.

Per writer: the documents and the relevant documents that only the other names reach, the queries that lose
something, and set recall and F1 of both runs, computed as in eval/metrics.py (an empty set has recall 0 and
F1 0; a query with no relevant document in the qrels, or without a formula, is left out). Plus the queries with
the most relevant documents lost, with the other names their formula used.

Usage: python eval/stale_names.py --work WORK --qrels WORK/qrels/qrels_ai.txt --name ai
           [--tags graph_annotator_a_cr,graph_annotator_b_cr]
Needs WORK/runs/<tag>_sets.jsonl and <tag>_current_sets.jsonl for each tag.
Outputs: WORK/eval/stale_names_<name>.json and .md
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonl import read_jsonl, write_json  # noqa: E402
from common.trec import read_qrels  # noqa: E402


def set_scores(docs: set[str], rel: set[str]) -> tuple[float, float]:
    """(recall, F1) of a set against the relevant documents; an empty set has recall 0 and F1 0."""
    hits = len(docs & rel)
    r = hits / len(rel)
    p = hits / len(docs) if docs else 0.0
    return r, (2 * p * r / (p + r) if p + r else 0.0)


def compare(main_rows: list[dict], cur_rows: list[dict], qrels: dict[str, dict[str, int]]) -> dict:
    cur = {r["query_id"]: r for r in cur_rows}
    per_q, not_subset = [], []
    for m in main_rows:
        qid = m["query_id"]
        if m.get("note") == "no formula" or qid not in cur:
            continue
        a, b = set(m.get("doc_ids") or []), set(cur[qid].get("doc_ids") or [])
        if not b <= a:
            not_subset.append(qid)
        rel = {d for d, v in (qrels.get(qid) or {}).items() if v > 0}
        judged = set(qrels.get(qid) or {})
        lost = a - b
        row = {"query_id": qid, "B_main": len(a), "B_current": len(b), "docs_lost": len(lost),
               "docs_lost_unjudged": len(lost - judged), "relevant_lost": len(lost & rel), "n_relevant": len(rel),
               "other_names": sorted(m.get("alias_terms") or [])}
        if rel:
            row["recall_main"], row["f1_main"] = set_scores(a, rel)
            row["recall_current"], row["f1_current"] = set_scores(b, rel)
        per_q.append(row)
    scored = [r for r in per_q if r["n_relevant"]]

    def mean(key):
        return round(statistics.mean(r[key] for r in scored), 4) if scored else None
    return {
        "queries": len(per_q), "queries_with_relevant": len(scored),
        "queries_losing_documents": sum(1 for r in per_q if r["docs_lost"]),
        "queries_losing_relevant": sum(1 for r in per_q if r["relevant_lost"]),
        "documents_lost": sum(r["docs_lost"] for r in per_q),
        "documents_lost_unjudged": sum(r["docs_lost_unjudged"] for r in per_q),
        "relevant_lost": sum(r["relevant_lost"] for r in per_q),
        "relevant_found_main": sum(len(set(m.get("doc_ids") or []) & {d for d, v in (qrels.get(m["query_id"]) or {}).items() if v > 0})
                                   for m in main_rows if m.get("note") != "no formula"),
        "mean_recall_main": mean("recall_main"), "mean_recall_current": mean("recall_current"),
        "mean_f1_main": mean("f1_main"), "mean_f1_current": mean("f1_current"),
        "empty_sets_main": sum(1 for r in per_q if r["B_main"] == 0),
        "empty_sets_current": sum(1 for r in per_q if r["B_current"] == 0),
        "not_subset": not_subset, "per_query": per_q,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--qrels", required=True, help="the AI judge's qrels (qrels_ai.txt)")
    ap.add_argument("--name", default="ai")
    ap.add_argument("--tags", default="graph_annotator_a_cr,graph_annotator_b_cr")
    ap.add_argument("--top", type=int, default=10, help="queries listed with the most relevant documents lost")
    args = ap.parse_args()
    work = Path(args.work)
    qrels = read_qrels(args.qrels)
    out = {"qrels": str(args.qrels), "writers": {}}
    lines = ["# What the archive's old entity names cost the graph", "",
             f"qrels: {args.qrels} (the AI judge's labels). Main run: every name of a page; current: today's pages only.",
             "An empty set has recall 0 and F1 0; queries without a relevant document or without a formula are left out of the means.", "",
             "| writer | queries | losing documents | losing relevant | documents lost (unjudged) | relevant lost / found by the main run "
             "| recall main / current | F1 main / current | empty sets main / current |",
             "|---|---:|---:|---:|---:|---:|---|---|---|"]
    for tag in [t.strip() for t in args.tags.split(",") if t.strip()]:
        runs = work / "runs"
        main_p, cur_p = runs / f"{tag}_sets.jsonl", runs / f"{tag}_current_sets.jsonl"
        for p in (main_p, cur_p):
            if not p.exists():
                raise SystemExit(f"missing {p}: run systems/graph_boolean.py" + (" --current-names-only" if p == cur_p else ""))
        res = compare(list(read_jsonl(main_p)), list(read_jsonl(cur_p)), qrels)
        out["writers"][tag] = res
        lines.append(f"| {tag} | {res['queries']} | {res['queries_losing_documents']} | {res['queries_losing_relevant']} | "
                     f"{res['documents_lost']} ({res['documents_lost_unjudged']}) | {res['relevant_lost']} / {res['relevant_found_main']} | "
                     f"{res['mean_recall_main']} / {res['mean_recall_current']} | {res['mean_f1_main']} / {res['mean_f1_current']} | "
                     f"{res['empty_sets_main']} / {res['empty_sets_current']} |")
        if res["not_subset"]:
            print(f"WARNING {tag}: the current-names set is not inside the main set for {len(res['not_subset'])} queries "
                  f"(a formula with NOT): {', '.join(res['not_subset'][:10])}", file=sys.stderr)
    for tag, res in out["writers"].items():
        top = sorted((r for r in res["per_query"] if r["relevant_lost"]), key=lambda r: (-r["relevant_lost"], r["query_id"]))[:args.top]
        if top:
            lines += ["", f"## {tag}: the queries with the most relevant documents lost", "",
                      "| query | relevant lost | documents lost | B main / current | other names in the formula |", "|---|---:|---:|---|---|"]
            lines += [f"| {r['query_id']} | {r['relevant_lost']} | {r['docs_lost']} | {r['B_main']} / {r['B_current']} | {', '.join(r['other_names'])} |" for r in top]
    (work / "eval").mkdir(parents=True, exist_ok=True)
    write_json(work / "eval" / f"stale_names_{args.name}.json", out)
    (work / "eval" / f"stale_names_{args.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
