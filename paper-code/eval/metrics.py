"""
metrics.py -- the numbers of the paper, computed from the runs and one qrels file (the paper,
Section 5).

Run it once per qrels file: qrels_ai.txt (the main tables, all queries), qrels_human.txt and qrels_ai_check.txt (the
check queries, human labels and the AI's labels on the same documents), and each again with --before 2025-04.

THE RANKED SYSTEMS (--ranked: BM25 and the pipeline; --generous: the generous version of each writer's formula;
  --extra: other ranked runs reported as rows without any test, such as bm25_claimart and pipeline_claimart, the text
  systems on the claim plus the explanation, a second row next to the main runs on the explanation alone):
  P@k, nDCG@k and R@k for k in --cutoffs (10, 20), and R@--depth (recall against the pool: the relevant documents of
  the qrels are the denominator), computed with ir_measures; tie-aware P@k (the expected precision when tied scores
  are in any order); judged@k, the share of the top k that has a label. A document without a label counts as NOT
  relevant (standard pooling practice); judged@k shows how often that happens.
THE STRICT SETS (--sets: one graph run per writer), measured as sets (Turtle 1994): precision, recall and F1 of the
  whole set, its size B, the judged share, and the precision over the judged members only. An EMPTY set (B = 0) has
  recall 0 and F1 0; its precision is undefined and not averaged, and the number of empty sets is reported.
  The two writers: per query, the mean of the writers' values is the graph's score; the better writer per query
  (highest F1) is reported only as a ceiling, since picking it needs the labels.
THE LISTS AT THE SET'S SIZE (TREC Legal): every ranked system cut at each writer's B for that query: P@B, R@B, F1@B,
  judged@B; a B larger than the run's length (100) is compared at 100 (precision over the 100 documents
  returned) and counted. With B = 0 nothing is compared
  (P@B undefined, R@B and F1@B 0, like the empty set).
PAIRED COMPARISONS, per query: wins, losses, ties, mean difference and the Wilcoxon signed-rank test, with the
  Benjamini-Hochberg correction over every test of the table: BM25 against the pipeline on P@k, nDCG@k and R@depth;
  the graph (mean of the writers) against each ranked system at the same size (set precision against P@B, set recall
  against R@B, set F1 against F1@B).
POOL COMPLETENESS (Zobel): relevant documents in the top d of each ranked system and of their union, d = 10, 20, ...
  up to --depth.
Queries with no relevant document in the qrels are excluded from every mean and test and listed.
--before YYYY-MM keeps only the documents published before that month or undated, in the qrels, the runs and the
  sets, before scoring (the part of the archive where entity linking worked; the month is the one
  stats/plot_archive.py finds, \\entityStopMonth). A list cut this way moves up documents from below the judged depth;
  pool/extend_pool_before.py adds them to the pool with the same month, so that they are judged: judged@k shows
  whether any is still missing.

Usage:
  python eval/metrics.py --work WORK --qrels WORK/qrels/qrels_ai.txt --ranked bm25,pipeline \
      --generous relaxed_annotator_a,relaxed_annotator_b --sets graph_annotator_a_cr,graph_annotator_b_cr --name ai [--before 2025-04]
Outputs: WORK/eval/metrics_<name>.json and metrics_<name>.md
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import kg  # noqa: E402
from common.jsonl import read_jsonl, write_json  # noqa: E402
from common.trec import read_run, read_qrels  # noqa: E402


def names(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return round(statistics.mean(xs), 4) if xs else None


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank_from_top, i in enumerate(reversed(order)):
        k = m - rank_from_top
        val = min(prev, pvals[i] * m / k)
        adj[i] = val
        prev = val
    return adj


def f1(p, r):
    if p is None or r is None:
        return None
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def ranked_measures(run: dict, qrels: dict, cutoffs: list[int], depth: int, qids: list[str]):
    import ir_measures
    from ir_measures import P, nDCG, R
    ks = sorted(set(cutoffs) | {depth})
    measures = [P @ k for k in cutoffs] + [nDCG @ k for k in cutoffs] + [R @ k for k in ks]
    q_objs = [ir_measures.Qrel(q, d, rel) for q, docs in qrels.items() for d, rel in docs.items()]
    r_objs = [ir_measures.ScoredDoc(q, d, s) for q, docs in run.items() for d, s in docs]
    per_q: dict[str, dict[str, float]] = {q: {} for q in qids}
    for m in ir_measures.iter_calc(measures, q_objs, r_objs):
        if m.query_id in per_q:
            per_q[m.query_id][str(m.measure)] = m.value
    for q in qids:
        for m in measures:
            per_q[q].setdefault(str(m), 0.0)
    for q in qids:
        docs = [d for d, _ in run.get(q, [])]
        for k in cutoffs:
            top = docs[:k]
            per_q[q][f"judged@{k}"] = (sum(1 for d in top if d in qrels.get(q, {})) / len(top)) if top else None
    return per_q, [str(m) for m in measures]


def tie_aware_precision(ordered: list[str], tier_key, qrels_q: dict[str, int], k: int) -> float:
    """Expected precision at k when documents inside a tier (same score) are in an arbitrary order: a tier that
    straddles rank k contributes its relevant count times the share of its slots that fall inside the top k.
    Equal to P@k when there are no ties."""
    if k <= 0 or not ordered:
        return 0.0
    expected, slots = 0.0, k
    i = 0
    while i < len(ordered) and slots > 0:
        key = tier_key(ordered[i])
        j = i
        while j < len(ordered) and tier_key(ordered[j]) == key:
            j += 1
        tier = ordered[i:j]
        n_rel = sum(1 for d in tier if qrels_q.get(d) == 1)
        if slots >= len(tier):
            expected += n_rel
            slots -= len(tier)
        else:
            expected += n_rel * slots / len(tier)
            slots = 0
        i = j
    return expected / k


def set_measures(sets: dict, qrels: dict, qids: list[str]):
    out = {}
    for q in qids:
        if q not in sets:
            continue  # the writer wrote no formula for this query
        ids = sets[q]
        rel = {d for d, r in qrels.get(q, {}).items() if r == 1}
        judged = [d for d in ids if d in qrels.get(q, {})]
        hits = sum(1 for d in ids if d in rel)
        B = len(ids)
        p = hits / B if B else None
        r = hits / len(rel) if rel else None
        out[q] = {"B": B, "hits": hits, "set_P": p, "set_R": r, "set_F1": f1(p, r) if B else (0.0 if rel else None),
                  "judged_share": (len(judged) / B) if B else None, "set_P_judged_only": (hits / len(judged)) if judged else None}
    return out


def at_B(run: dict, qrels: dict, Bs: dict, qids: list[str]):
    out = {}
    for q in qids:
        if q not in Bs:
            continue
        B = Bs[q]
        docs = [d for d, _ in run.get(q, [])]
        rel = {d for d, r in qrels.get(q, {}).items() if r == 1}
        if B == 0:
            out[q] = {"P@B": None, "R@B": 0.0 if rel else None, "F1@B": 0.0 if rel else None, "judged@B": None, "note": "B=0"}
            continue
        # A set larger than the run (100 documents) is compared at the run's length: precision over the
        # documents the list returned, not over B, which would count documents the list never returned as misses.
        top = docs[:B]
        hits = sum(1 for d in top if d in rel)
        p, r = (hits / len(top) if top else None), ((hits / len(rel)) if rel else None)
        out[q] = {"P@B": p, "R@B": r, "F1@B": f1(p, r), "judged@B": (sum(1 for d in top if d in qrels.get(q, {})) / len(top)) if top else None,
                  "note": "B>run length" if B > len(docs) else ""}
    return out


def zobel_curve(runs: dict[str, dict], qrels: dict, qids: list[str], depths: list[int]):
    curve = {}
    for name, run in runs.items():
        curve[name] = [sum(sum(1 for doc, _ in run.get(q, [])[:d] if qrels.get(q, {}).get(doc) == 1) for q in qids) for d in depths]
    curve["union"] = []
    for d in depths:
        found = 0
        for q in qids:
            u = set()
            for run in runs.values():
                u.update(doc for doc, _ in run.get(q, [])[:d])
            found += sum(1 for doc in u if qrels.get(q, {}).get(doc) == 1)
        curve["union"].append(found)
    return curve


def paired(xa: list, xb: list) -> dict:
    from scipy.stats import wilcoxon
    keep = [(x, y) for x, y in zip(xa, xb) if x is not None and y is not None]
    diffs = [x - y for x, y in keep]
    wins, losses = sum(1 for d in diffs if d > 0), sum(1 for d in diffs if d < 0)
    p = float(wilcoxon(diffs, zero_method="wilcox").pvalue) if any(d != 0 for d in diffs) else 1.0
    return {"n": len(diffs), "mean_diff": round(statistics.mean(diffs), 4) if diffs else None, "wins": wins, "losses": losses,
            "ties": len(diffs) - wins - losses, "p_wilcoxon": round(p, 5)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--ranked", default="bm25,pipeline")
    ap.add_argument("--generous", default="", help="the generous version of each writer's formula (ranked runs)")
    ap.add_argument("--extra", default="", help="other ranked runs reported as rows, without tests (e.g. bm25_claimart,pipeline_claimart)")
    ap.add_argument("--sets", default="graph", help="the strict result sets, one graph run per writer (WORK/runs/<tag>_sets.jsonl)")
    ap.add_argument("--cutoffs", default="10,20")
    ap.add_argument("--depth", type=int, default=20, help="recall against the pool at this depth (the judged depth)")
    ap.add_argument("--before", default=None, help="YYYY-MM: keep only documents published before this month, or undated")
    ap.add_argument("--name", required=True, help="e.g. ai, human, ai_check, ai_before")
    args = ap.parse_args()
    work = Path(args.work)
    cutoffs = [int(x) for x in args.cutoffs.split(",")]
    qrels = read_qrels(args.qrels)
    ranked = [s for s in names(args.ranked) + names(args.generous) + names(args.extra) if (work / "runs" / f"{s}.trec").exists()]
    main_ranked = [s for s in names(args.ranked) if s in ranked]
    runs = {s: read_run(work / "runs" / f"{s}.trec") for s in ranked}
    set_tags = [s for s in names(args.sets) if (work / "runs" / f"{s}_sets.jsonl").exists()]
    missing = [s for s in names(args.ranked) + names(args.generous) + names(args.extra) if s not in ranked] + [s for s in names(args.sets) if s not in set_tags]
    if missing:
        print(f"WARNING: no run file for {', '.join(missing)} in {work / 'runs'}: left out of the tables (a misspelt name?)")
    # A query where the writer has no formula (a row of the sheet that did not import: graph_boolean.py writes it with
    # formula None and note 'no formula') is left out of that writer's set measures, never counted as an empty set: an
    # empty set measures the graph, a missing formula is a data problem to repair before the final run.
    sets, no_formula = {}, {}
    for s in set_tags:
        rows = list(read_jsonl(work / "runs" / f"{s}_sets.jsonl"))
        sets[s] = {r["query_id"]: r["doc_ids"] for r in rows if r.get("formula")}
        no_formula[s] = sorted(r["query_id"] for r in rows if not r.get("formula"))
        if no_formula[s]:
            print(f"WARNING: {s}: {len(no_formula[s])} queries have no formula and are left out of its set measures "
                  f"(repair them before the final run): {', '.join(no_formula[s][:12])}{' ...' if len(no_formula[s]) > 12 else ''}")
    dates = {}
    if (work / "corpus.jsonl").exists():
        from systems.graph_boolean import sortable_date  # same date rule as the graph system
        dates = {d["cr_iri"]: sortable_date(d.get("date_published")) for d in kg.load_corpus(work / "corpus.jsonl")}
    removed = 0
    if args.before:
        keep = lambda d: not dates.get(d) or dates[d][:7] < args.before  # noqa: E731
        removed = sum(1 for docs in qrels.values() for d in docs if not keep(d))
        qrels = {q: {d: v for d, v in docs.items() if keep(d)} for q, docs in qrels.items()}
        runs = {s: {q: [(d, sc) for d, sc in docs if keep(d)] for q, docs in run.items()} for s, run in runs.items()}
        sets = {s: {q: [d for d in ids if keep(d)] for q, ids in st.items()} for s, st in sets.items()}
    qids_all = sorted({q for r in runs.values() for q in r} | set(qrels) | {q for st in sets.values() for q in st})
    n_rel = {q: sum(qrels.get(q, {}).values()) for q in qids_all}
    excluded = [q for q in qids_all if n_rel[q] == 0 and q in qrels]
    qids = [q for q in qids_all if n_rel[q] > 0]
    if not qids:
        raise SystemExit("no query has a relevant document in these qrels")

    per_q: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    names_ranked = []
    for s in ranked:
        pq, names_ranked = ranked_measures(runs[s], qrels, cutoffs, args.depth, qids)
        for q in qids:
            docs = runs[s].get(q, [])
            sc = {d: round(v, 9) for d, v in docs}
            for k in cutoffs:
                pq[q][f"P@{k} tie-aware"] = tie_aware_precision([d for d, _ in docs], lambda d, sc=sc: sc[d], qrels.get(q, {}), k)
        for w in set_tags:
            atb = at_B(runs[s], qrels, {q: len(v) for q, v in sets[w].items()}, qids)
            for q, v in atb.items():
                pq[q].update({f"{m}[{w}]": v[m] for m in ("P@B", "R@B", "F1@B", "judged@B")})
                pq[q][f"B>run length[{w}]"] = v["note"] == "B>run length"
        per_q[s] = pq
        cols = names_ranked + [f"P@{k} tie-aware" for k in cutoffs] + [f"judged@{k}" for k in cutoffs]
        summary[s] = {m: mean(pq[q][m] for q in qids) for m in cols}
        for w in set_tags:
            for m in ("P@B", "R@B", "F1@B", "judged@B"):
                summary[s][f"{m}[{w}]"] = mean(pq[q].get(f"{m}[{w}]") for q in qids)
            summary[s][f"queries with B>run length[{w}]"] = sum(1 for q in qids if pq[q].get(f"B>run length[{w}]"))
    for w in set_tags:
        sm = set_measures(sets[w], qrels, qids)
        per_q[w] = sm
        summary[w] = {"set_P": mean(v["set_P"] for v in sm.values()), "set_R": mean(v["set_R"] for v in sm.values()),
                      "set_F1": mean(v["set_F1"] for v in sm.values()), "judged_share": mean(v["judged_share"] for v in sm.values()),
                      "set_P_judged_only": mean(v["set_P_judged_only"] for v in sm.values()),
                      "queries": len(sm), "queries_B_zero": sum(1 for v in sm.values() if v["B"] == 0),
                      "median_B": statistics.median([v["B"] for v in sm.values()]) if sm else None}
    # the two writers: the mean per query is the graph's score; the better writer per query is only a ceiling
    graph_mean, graph_best = {}, {}
    if set_tags:
        for q in qids:
            rows = [per_q[w][q] for w in set_tags if q in per_q[w]]
            if not rows:
                continue
            graph_mean[q] = {m: mean(r[m] for r in rows) for m in ("set_P", "set_R", "set_F1")}
            best = max(rows, key=lambda r: (r["set_F1"] or 0.0))
            graph_best[q] = {m: best[m] for m in ("set_P", "set_R", "set_F1")}
            for s in main_ranked:
                for m in ("P@B", "R@B", "F1@B"):
                    graph_mean[q][f"{s} {m}"] = mean(per_q[s][q].get(f"{m}[{w}]") for w in set_tags if q in per_q[w])
        summary["graph (mean of the writers)"] = {m: mean(v[m] for v in graph_mean.values()) for m in ("set_P", "set_R", "set_F1")}
        summary["graph (better writer per query: a ceiling)"] = {m: mean(v[m] for v in graph_best.values()) for m in ("set_P", "set_R", "set_F1")}

    # paired comparisons
    tests = {}
    k_main = cutoffs[-1]
    if len(main_ranked) >= 2:
        for a, b in combinations(main_ranked, 2):
            for m in (f"P@{cutoffs[0]}", f"P@{k_main}", f"nDCG@{k_main}", f"R@{args.depth}"):
                tests[f"{a} vs {b}: {m}"] = paired([per_q[a][q][m] for q in qids], [per_q[b][q][m] for q in qids])
    for s in main_ranked:
        for gm, lm in (("set_P", "P@B"), ("set_R", "R@B"), ("set_F1", "F1@B")):
            qs = [q for q in qids if q in graph_mean]
            if qs:
                tests[f"graph (writers' mean) vs {s}: {gm} vs {lm}"] = paired([graph_mean[q][gm] for q in qs], [graph_mean[q][f"{s} {lm}"] for q in qs])
    if tests:
        adj = benjamini_hochberg([v["p_wilcoxon"] for v in tests.values()])
        for (k, v), pa in zip(tests.items(), adj):
            v["p_BH"] = round(pa, 5)
    depths = list(range(10, args.depth + 1, 10))
    curve = zobel_curve({s: runs[s] for s in main_ranked}, qrels, qids, depths)

    report = {"qrels": str(args.qrels), "before": args.before, "queries_without_formula": no_formula, "qrels_pairs_removed_by_before": removed, "n_queries": len(qids),
              "excluded_queries_without_relevant": excluded, "total_relevant": sum(n_rel[q] for q in qids), "cutoffs": cutoffs,
              "depth": args.depth, "unjudged_policy": "unjudged documents count as not relevant; judged@k shows the judged share",
              "summary": summary, "tests": tests, "zobel_depths": depths, "zobel_curve": curve, "per_query": per_q,
              "graph_mean_per_query": graph_mean, "graph_best_per_query": graph_best, "n_relevant_per_query": n_rel}
    out = work / "eval"
    write_json(out / f"metrics_{args.name}.json", report)
    md = [f"# Metrics ({args.name}: {Path(args.qrels).name}" + (f", documents before {args.before} or undated" if args.before else "") + ")", "",
          f"queries measured: {len(qids)}; excluded because judging found no relevant document: {len(excluded)}"
          f"{' (' + ', '.join(excluded) + ')' if excluded else ''}; relevant documents: {report['total_relevant']}",
          "unjudged documents count as not relevant; judged@k = share of the top k that has a label; tie-aware = expected precision when tied scores are in any order", ""]
    cols = names_ranked + [f"P@{k} tie-aware" for k in cutoffs] + [f"judged@{k}" for k in cutoffs]
    md += ["## Ranked systems", "", "| system | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    for s in ranked:
        md.append(f"| {s} | " + " | ".join(fmt(summary[s].get(c)) for c in cols) + " |")
    if set_tags:
        md += ["", "## The graph: strict sets, measured as sets (an empty set: recall 0, F1 0, precision undefined)", "",
               "queries left out because the writer has no formula: " + "; ".join(f"{w} {len(no_formula[w])}" for w in set_tags), "",
               "| run | queries | empty sets | median B | set precision | set recall | set F1 | judged share | precision on judged only |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for w in set_tags:
            g = summary[w]
            md.append(f"| {w} | {g['queries']} | {g['queries_B_zero']} | {g['median_B']} | {fmt(g['set_P'])} | {fmt(g['set_R'])} | {fmt(g['set_F1'])} | "
                      f"{fmt(g['judged_share'])} | {fmt(g['set_P_judged_only'])} |")
        for lab in ("graph (mean of the writers)", "graph (better writer per query: a ceiling)"):
            g = summary[lab]
            md.append(f"| {lab} | {len(graph_mean)} | | | {fmt(g['set_P'])} | {fmt(g['set_R'])} | {fmt(g['set_F1'])} | | |")
        md += ["", "## The ranked systems cut at each writer's set size B", "",
               "| system | " + " | ".join(f"{m} [{w}]" for w in set_tags for m in ("P@B", "R@B", "F1@B", "judged@B")) + " |",
               "|---|" + "---:|" * (4 * len(set_tags))]
        for s in main_ranked:
            md.append(f"| {s} | " + " | ".join(fmt(summary[s].get(f"{m}[{w}]")) for w in set_tags for m in ("P@B", "R@B", "F1@B", "judged@B")) + " |")
        md.append("queries where B is larger than the run (the whole run used): " +
                  "; ".join(f"{s} [{w}] {summary[s][f'queries with B>run length[{w}]']}" for s in main_ranked for w in set_tags))
    if tests:
        md += ["", "## Paired comparisons per query (Wilcoxon signed-rank; p (BH) corrected over every test of this table)", "",
               "| comparison | n | mean diff | wins | losses | ties | p | p (BH) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for k, v in tests.items():
            md.append(f"| {k} | {v['n']} | {v['mean_diff']} | {v['wins']} | {v['losses']} | {v['ties']} | {v['p_wilcoxon']} | {v['p_BH']} |")
    md += ["", "## Pool depth curve (Zobel): relevant documents in the top d, summed over queries", "",
           "| depth | " + " | ".join(curve) + " |", "|---:|" + "---:|" * len(curve)]
    for i, d in enumerate(depths):
        md.append(f"| {d} | " + " | ".join(str(curve[s][i]) for s in curve) + " |")
    (out / f"metrics_{args.name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
