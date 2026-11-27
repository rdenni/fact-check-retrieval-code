"""
oracle_formula.py -- the oracle ceiling of the graph (written before any
relevance label exists).

The question it answers: is the graph's result due to WHO wrote the formulas, or to
the entities themselves? After judging, the relevant fact-checks of every query are
known. This script cheats on purpose: it uses those answers to build, one term at a
time, the best Boolean formula the archive's entities allow, and runs it on the whole
archive. If even this formula does badly, the writers cannot be blamed. The ranked
systems get the same cheat: the best place to cut their list.

Two formula families (a "term" is what is joined with OR):
  singles   OR of single entities              e1 OR e2 OR e3 ...
  pairs     OR of terms that are one entity or the AND of two entities
            e1 OR (e2 AND e3) OR ...  (closer to the writers' AND formulas; can overfit more)
Candidates: only entities (and pairs of entities) that occur in at least one relevant
fact-check; a term that adds no new relevant fact-check is never taken (it cannot raise F1).

Greedy: start empty; at each step add the candidate that gives the highest F1; stop
when no candidate raises F1 or after --k-max terms. Ties: the smaller set, then the
alphabetical order of the term, so the run is reproducible. Greedy is not guaranteed
to find the best formula of a given size, so the ceiling is itself a lower bound on
what the entities allow.

Fact-checks nobody judged. A formula runs on the whole archive and returns documents
outside the pool. Two treatments, both reported, each with its own greedy run:
  strict        unjudged = not relevant (the usual TREC rule). Precision can only be
                too low, so this ceiling is pessimistic: the true ceiling of the
                formula found is at least this.
  judged_only   precision on the judged documents of the set only (the condensed-list
                idea). Optimistic: unjudged documents are usually less often relevant
                than judged ones, which came from the top of some system.
Recall is always against the relevant documents found by judging (never absolute).

Two budgets for the number of terms:
  K = the number of distinct entities in the writer's formula for that query (the same
      size budget as the writer; from the graph run's set file, capped at --k-max);
  K = 1 ... --k-max, as a curve: how much each extra term helps.

The text side: for every ranked run, the cutoff k (1 to the run's depth) with the
highest F1@k, under the same two treatments (in practice within the judged depth, 20 or
the set's size B: below it a document is judged only if another system brought it into
the pool, and an unjudged one cannot raise F1).
Two writers: run the script once per writer (--graph-tag graph_annotator_a_cr, then
graph_annotator_b_cr, each with its own --name), so that each writer's budget is her own
number of pages.

THE MEMORISATION CONTROL (written before any label): with enough terms an
entity formula can pick out ANY small set of documents, one rare page per document
(on the real archive, 10 single pages found 10 random fact-checks with F1 0.86, 10 pairs
with F1 1.0). So for every query the same greedy oracle (singles, strict) is also run on
a random set of fact-checks of the same size as the relevant set, drawn from the whole
archive with the seed "<--control-seed>:<query_id>". Its F1 is the score a formula gets
by memorising documents that share no narrative. The oracle on the real relevant set
counts as evidence that the entities describe the narrative only by how much it exceeds
this control.

THE PRIMARY NUMBER (fixed before the run): the singles oracle at the writer's size,
strict, next to its control and to the pipeline's best cutoff (strict), compared per query
(mean difference, wins and losses, Wilcoxon signed-rank). "Clearly above" means a mean
difference above zero with p < 0.05. The other rows (judged_only, pairs, up to --k-max
terms) are context.

THE READING, WRITTEN BEFORE THE RUN, on the primary number:
  (a) the graph's oracle is below the text pipeline's best cutoff, or not clearly above its
      memorisation control: the entities cannot do this task even with perfect knowledge,
      whoever writes the formula;
  (b) the graph's oracle is well above the writers' formulas and near the text pipeline:
      the entities could work, the formulas are hard to write ("Boolean entity search is
      hard to use"); reported as such.

Inputs: WORK/entities_<node>.jsonl, WORK/runs/<graph-tag>_sets.jsonl (optional: the writer's
formula size and F1), WORK/runs/<system>.trec, the qrels.
Outputs: WORK/eval/oracle_<name>.md (tables and the reading), WORK/eval/oracle_<name>.jsonl
(per query: the formulas found, their terms as Wikipedia titles, the curves).

Usage: python eval/oracle_formula.py --work WORK --qrels WORK/qrels/qrels_ai.txt --name ai_a --node cr --graph-tag graph_annotator_a_cr
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.parse
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import kg  # noqa: E402
from common.jsonl import read_jsonl, write_jsonl  # noqa: E402
from common.trec import read_qrels, read_run  # noqa: E402

TREATMENTS = ("strict", "judged_only")
FAMILIES = ("singles", "pairs")


def title(uri: str) -> str:
    tail = uri.split("/resource/", 1)[1] if "/resource/" in uri else uri.rsplit("/", 1)[-1]  # 'HIV/AIDS' keeps its slash
    return urllib.parse.unquote(tail).replace("_", " ")


def f1(hits: int, size: int, judged: int, n_rel: int, treatment: str) -> float:
    """F1 of a set with `hits` relevant members, `size` members, `judged` judged members."""
    denom = size if treatment == "strict" else judged
    if hits == 0 or denom == 0 or n_rel == 0:
        return 0.0
    p, r = hits / denom, hits / n_rel
    return 2 * p * r / (p + r)


def candidates(family: str, rel: set[str], doc_ents: dict[str, set[str]], inv: dict[str, set[str]]) -> dict[str, set[str]]:
    """term label -> set of documents. Singles: entities of relevant documents; pairs: also every two
    entities that occur together in a relevant document (the AND of the two)."""
    ents = set()
    for d in rel:
        ents |= doc_ents.get(d, set())
    out = {f'"{title(e)}"': inv.get(e, set()) for e in ents}
    if family == "pairs":
        for d in rel:
            for a, b in combinations(sorted(doc_ents.get(d, set())), 2):
                key = f'("{title(a)}" AND "{title(b)}")'
                if key not in out:
                    sa, sb = inv.get(a, set()), inv.get(b, set())
                    out[key] = sa & sb if len(sa) <= len(sb) else sb & sa
    return out


def greedy(cands: dict[str, set[str]], rel: set[str], judged: set[str], k_max: int, treatment: str) -> dict:
    """Add one term at a time, the one with the highest F1; stop when nothing raises F1."""
    chosen: list[str] = []
    S: set[str] = set()
    hits = size = n_jud = 0
    best = 0.0
    curve: list[float] = []
    states: list[dict] = []
    for _ in range(k_max):
        top = None
        for key in sorted(cands):
            c = cands[key]
            new = c - S
            if not (new & rel):
                continue  # no new relevant document: cannot raise F1
            h = hits + len(new & rel)
            s = size + len(new)
            j = n_jud + len(new & judged)
            score = f1(h, s, j, len(rel), treatment)
            if top is None or score > top[0] + 1e-12 or (abs(score - top[0]) <= 1e-12 and s < top[2]):
                top = (score, key, s, h, j)
        if top is None or top[0] <= best + 1e-12:
            break
        best, key, size, hits, n_jud = top
        S |= cands[key]
        chosen.append(key)
        curve.append(best)
        states.append({"F1": best, "B": size, "hits": hits, "judged": n_jud})
    while len(curve) < k_max:  # stopped early: the curve stays flat
        curve.append(best)
    return {"terms": chosen, "curve": curve, "states": states}


def at_budget(g: dict, k: int | None) -> dict | None:
    """The greedy formula cut at k terms (or fewer, if greedy stopped before k)."""
    if not k:
        return None
    n = min(k, len(g["terms"]))
    if n == 0:
        return {"F1": 0.0, "B": 0, "terms": [], "formula": ""}
    st = g["states"][n - 1]
    return {"F1": st["F1"], "B": st["B"], "hits": st["hits"], "judged": st["judged"],
            "terms": g["terms"][:n], "formula": " OR ".join(g["terms"][:n])}


def best_cutoff(docs: list[str], rel: set[str], judged: set[str], treatment: str) -> dict:
    best = {"F1": 0.0, "k": 0}
    hits = jud = 0
    for k, d in enumerate(docs, 1):
        hits += d in rel
        jud += d in judged
        score = f1(hits, k, jud, len(rel), treatment)
        if score > best["F1"] + 1e-12:
            best = {"F1": score, "k": k}
    return best


def set_f1(ids: set[str], rel: set[str], judged: set[str], treatment: str) -> float:
    return f1(len(ids & rel), len(ids), len(ids & judged), len(rel), treatment)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x):
    return "-" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--name", required=True, help="human or full")
    ap.add_argument("--node", default="cr", choices=kg.NODE_CHOICES, help="the entity index the graph run used (the paper: cr)")
    ap.add_argument("--graph-tag", default="graph", help="WORK/runs/<tag>_sets.jsonl: the writer's formula, for its size and its F1")
    ap.add_argument("--systems", default="bm25,pipeline", help="ranked runs that get the best-cutoff oracle")
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--control-seed", default="20261111", help="seed of the random sets of the memorisation control")
    args = ap.parse_args()
    work = Path(args.work)
    families = [f for f in args.families.split(",") if f]
    qrels = read_qrels(args.qrels)
    doc_ents = kg.load_entity_index(work / f"entities_{args.node}.jsonl")
    inv = kg.invert_entity_index(doc_ents)
    with open(work / "corpus.jsonl", encoding="utf-8") as fh:  # every document, with or without entities
        all_docs = sorted(json.loads(line)["cr_iri"] for line in fh if line.strip())
    sets_path = work / "runs" / f"{args.graph_tag}_sets.jsonl"
    sets = {r["query_id"]: r for r in read_jsonl(sets_path)} if sets_path.exists() else {}
    runs = {}
    for s in [x for x in args.systems.split(",") if x]:
        p = work / "runs" / f"{s}.trec"
        if p.exists():
            runs[s] = read_run(p)
        else:
            print(f"(no run {p.name}; its best cutoff is skipped)")

    out_rows = []
    for qid in sorted(qrels):
        rel = {d for d, r in qrels[qid].items() if r == 1}
        if not rel:
            continue  # a query with no relevant document is excluded from the measures
        judged = set(qrels[qid])
        reachable = sum(1 for d in rel if doc_ents.get(d))
        s = sets.get(qid)
        k_writer = None
        writer = {}
        if s and s.get("formula"):
            own = {t: u for t, u in (s.get("terms") or {}).items() if t not in set(s.get("alias_terms") or [])}
            k_writer = min(len(set(own.values()) - {None}) or len(own), args.k_max) or None
            ids = set(s["doc_ids"])
            writer = {t: set_f1(ids, rel, judged, t) for t in TREATMENTS}
            writer["B"] = len(ids)
        row = {"query_id": qid, "n_relevant": len(rel), "n_relevant_with_entities": reachable,
               "max_recall_by_entities": reachable / len(rel), "writer_terms": k_writer, "writer": writer, "oracle": {}, "cutoff": {}}
        for fam in families:
            cands = candidates(fam, rel, doc_ents, inv)
            for t in TREATMENTS:
                g = greedy(cands, rel, judged, args.k_max, t)
                row["oracle"][f"{fam}/{t}"] = {"curve": g["curve"], "at_writer_size": at_budget(g, k_writer),
                                                "at_k_max": at_budget(g, args.k_max)}
        fake = set(random.Random(f"{args.control_seed}:{qid}").sample(all_docs, min(len(rel), len(all_docs))))
        g = greedy(candidates("singles", fake, doc_ents, inv), fake, fake, args.k_max, "strict")
        row["control_random"] = {"at_writer_size": (at_budget(g, k_writer) or {}).get("F1"), "at_k_max": at_budget(g, args.k_max)["F1"],
                                 "curve": g["curve"]}  # the control after 1 to k_max terms, to draw it next to the oracle's curve
        for sname, run in runs.items():
            docs = [d for d, _ in run.get(qid, [])]
            row["cutoff"][sname] = {t: best_cutoff(docs, rel, judged, t) for t in TREATMENTS}
        out_rows.append(row)

    out_dir = work / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / f"oracle_{args.name}.jsonl", out_rows)

    n = len(out_rows)
    lines = [f"# Oracle ceiling of the graph ({args.name} labels, node {args.node})", "",
             f"Queries with at least one relevant fact-check: {n}. F1 is averaged over them (a query whose formula finds nothing counts 0).",
             "strict: unjudged = not relevant (pessimistic). judged_only: precision on judged documents only (optimistic).", "",
             "| row | mean F1 strict | mean F1 judged_only | queries |", "|---|---|---|---|"]

    def col(get):
        """the two means, then the number of queries they average over (a row can skip a query: no writer formula)"""
        vals = {t: [get(r, t) for r in out_rows] for t in TREATMENTS}
        return [fmt(mean(vals[t])) for t in TREATMENTS] + [str(sum(1 for v in vals[TREATMENTS[0]] if v is not None))]

    if sets:
        lines.append("| the writer's formula | " + " | ".join(col(lambda r, t: r["writer"].get(t) if r["writer"] else 0.0)) + " |")
    for fam in families:
        lines.append(f"| oracle {fam}, as many terms as the writer used | " + " | ".join(
            col(lambda r, t, f=fam: (r["oracle"][f"{f}/{t}"]["at_writer_size"] or {}).get("F1") if r["writer_terms"] else None)) + " |")
        lines.append(f"| oracle {fam}, up to {args.k_max} terms | " + " | ".join(
            col(lambda r, t, f=fam: r["oracle"][f"{f}/{t}"]["at_k_max"]["F1"])) + " |")
    lines.append("| control: the singles oracle on a random set of the same size, as many terms as the writer used | " + " | ".join(
        col(lambda r, t: r["control_random"]["at_writer_size"] if t == "strict" and r["writer_terms"] else None)) + " |")
    lines.append(f"| control: the same, up to {args.k_max} terms | " + " | ".join(
        col(lambda r, t: r["control_random"]["at_k_max"] if t == "strict" else None)) + " |")
    for sname in runs:
        lines.append(f"| {sname}, best cutoff | " + " | ".join(col(lambda r, t, s=sname: r["cutoff"][s][t]["F1"])) + " |")
    lines += ["", f"Highest recall any entity formula can reach (relevant fact-checks that have entities at all), mean over queries: "
              f"{fmt(mean([r['max_recall_by_entities'] for r in out_rows]))}", ""]
    lines += ["## The curve: mean F1 after k terms, per family and treatment", "", "| k | " + " | ".join(f"{fam}/{t}" for fam in families for t in TREATMENTS) + " | control (singles/strict, random sets) |",
              "|---|" + "---|" * (len(families) * len(TREATMENTS) + 1)]
    for k in range(args.k_max):
        lines.append(f"| {k + 1} | " + " | ".join(fmt(mean([r["oracle"][f"{fam}/{t}"]["curve"][k] for r in out_rows]))
                                                  for fam in families for t in TREATMENTS)
                     + f" | {fmt(mean([r['control_random']['curve'][k] for r in out_rows]))} |")
    def paired(a_vals, b_vals, label):
        """per-query comparison on the queries where both values exist; Wilcoxon signed-rank as in eval/metrics.py"""
        pairs_ = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
        diffs = [a - b for a, b in pairs_]
        if not diffs:
            return f"{label}: no query with both values."
        wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        p = 1.0
        if any(d != 0 for d in diffs) and len(diffs) > 1:
            from scipy.stats import wilcoxon
            p = float(wilcoxon(diffs, zero_method="wilcox").pvalue)
        return (f"{label}: {len(diffs)} queries, mean difference {sum(diffs) / len(diffs):+.3f}, higher in {wins}, lower in {losses}, "
                f"Wilcoxon p = {p:.4g}")

    if "singles" in families:
        prim = [(r["oracle"]["singles/strict"]["at_writer_size"] or {}).get("F1") if r["writer_terms"] else None for r in out_rows]
        lines += ["", "## The primary number: singles oracle at the writer's size, strict",
                  "'Clearly above' means a mean difference above zero with Wilcoxon p < 0.05 (fixed before the run).", ""]
        lines.append("- " + paired(prim, [r["control_random"]["at_writer_size"] for r in out_rows], "oracle minus its memorisation control"))
        for sname in runs:
            lines.append("- " + paired(prim, [r["cutoff"][sname]["strict"]["F1"] for r in out_rows], f"oracle minus {sname}'s best cutoff"))
        if sets:
            lines.append("- " + paired(prim, [r["writer"].get("strict") if r["writer"] else 0.0 for r in out_rows], "oracle minus the writer's formula"))
    if runs and "singles" in families:
        ref = "pipeline" if "pipeline" in runs else next(iter(runs))
        for t in TREATMENTS:
            below = sum(1 for r in out_rows if r["oracle"][f"singles/{t}"]["at_k_max"]["F1"] < r["cutoff"][ref][t]["F1"])
            lines.append(f"\nQueries where the singles oracle ({t}, up to {args.k_max} terms) is below {ref}'s best cutoff: {below} of {n}.")
    lines += ["", "## The reading, written before the run",
              "(a) The graph's oracle is below the text pipeline's best cutoff, or not clearly above its memorisation control: the entities cannot do this task even with perfect knowledge, whoever writes the formula.",
              "(b) The graph's oracle is well above the writers' formulas and near the text pipeline: the entities could work, the formulas are hard to write; reported as such.",
              "The primary number is the singles oracle at the writer's size, strict, read next to its memorisation control and the pipeline's best cutoff; (a) also holds when the oracle is not clearly above the control ('clearly above': mean difference above zero, Wilcoxon p < 0.05 over the queries).",
              "", "Limits: the oracle knows only the judged documents (the pool), greedy search is not the exact optimum, and a formula of many terms can overfit; hence the curve and the writer-size budget. The ranked systems' best cutoff is in practice chosen within their judged depth (20, or the set's size B down to 100 when a strict set is larger: below it a document is judged only if another system brought it into the pool, and an unjudged one cannot raise F1), while an entity formula can return any number of documents: for a query with many relevant documents this caps the text side, so the comparison leans toward the graph."]
    lines += ["", "## Per query (singles, strict, up to the writer's size or k-max)", "", "| query | relevant | writer F1 | oracle F1 | oracle formula |", "|---|---|---|---|---|"]
    for r in out_rows:
        o = r["oracle"].get("singles/strict", {})
        pick = o.get("at_writer_size") or o.get("at_k_max") or {}
        lines.append(f"| {r['query_id']} | {r['n_relevant']} | {fmt(r['writer'].get('strict')) if r['writer'] else '-'} | {fmt(pick.get('F1'))} | {pick.get('formula', '')[:160]} |")
    (out_dir / f"oracle_{args.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / f'oracle_{args.name}.md'} and .jsonl ({n} queries)")


if __name__ == "__main__":
    main()
