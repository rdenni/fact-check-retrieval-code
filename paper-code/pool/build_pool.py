"""
build_pool.py -- which (query, document) pairs get judged, and which of them the two human judges check.

The judging plan: an AI judge labels every pair of
the pool; the two human judges label a check sample of 25 queries, only to measure how good the AI judge is. The pool
is built ONCE, after the runs are frozen, and nothing in it depends on a label.

THE POOL (every pair labelled by every AI judge), per query:
  - ranks 1 to --depth (20) of every ranked system (--ranked: BM25 and the pipeline), in trec_eval's order
    (common/trec.read_run: score descending, ties by document id descending);
  - every document of every strict result set (--sets: one graph run per writer, WORK/runs/<tag>_sets.jsonl),
    with no cap: a set is measured as a whole;
  - for every strict set of size B larger than --depth: the ranked systems down to min(B, the run's length), so that
    they can be compared with the set at the same size (TREC Legal; the runs stop at 100 documents);
  - ranks 1 to --depth of every AI-only ranked run (--ai-only: the generous version of each writer's formula).
THE CHECK SAMPLE (the human judges), written to check.json and assignments.json:
  - --check-queries (25) queries drawn uniformly at random with the seed --check-seed (20261116), in the order of the
    draw, which is the order the judges label them in; they stop after 15, 20 or 25 queries for reasons of time only;
  - for each check query: ranks 1 to --depth of the ranked systems (not the AI-only runs) and --set-sample (20)
    documents drawn at random from each strict set (the whole set when it has that many or fewer). The generous
    version is not in the check;
  - --shared (150) documents judged by BOTH judges, spread evenly over the first --shared-queries (15) check queries
    (one at a time from each, in turn, until the number is reached): their two labels, given independently, measure
    the judges' agreement (judge/adjudicate.py). There is no discussion: the final human label is the owner's;
  - every other check document is judged by one judge: each check query, in the order of the draw, goes to the judge
    with fewer documents so far (the first judge of --judges on a tie), so each query is read by one judge, as
    one TREC assessor judges one topic, and the two loads stay about equal. Only the counts decide, never a label.
Nobody sees which system returned a document: the judges' sheet and the AI prompt never show it.

Outputs (WORK/pool/):
  pool.jsonl        one line per pair: query_id, doc_id, ranks (system -> rank), min_rank, check (in the human
                    sample), claim_text, article_start (the first --context-chars characters of the explanation,
                    the fact-checker's text), date_published, author_name, country (the fact-checker's country from
                    --country-map, empty when unknown), rating (kept for analysis; never shown to a judge);
  check.json        the seed, the systems, the check queries in order, per query its documents, the set samples,
                    the owner judge and the shared documents;
  shared.json       the shared documents, [query_id, doc_id] each;
  assignments.json  per judge, the documents to label, in the order of the check queries;
  pool_summary.md   the counts.

Usage:
  python pool/build_pool.py --work WORK --queries Q.json --ranked bm25,pipeline \
      --sets graph_annotator_a_cr,graph_annotator_b_cr --ai-only relaxed_annotator_a,relaxed_annotator_b --judges R,M
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import kg  # noqa: E402
from common.jsonl import read_jsonl, write_jsonl, write_json  # noqa: E402
from common.queries import load_queries  # noqa: E402
from common.trec import read_run  # noqa: E402


def names(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def load_country_map(path: Path, required: bool) -> dict[str, str]:
    """publisher -> country (the fact-checker's IFCN country; snapshot/publisher_map.py). An absent default file gives
    no countries (the judges then see none); an absent file given on the command line stops the script."""
    import csv
    if not path.exists():
        if required:
            raise SystemExit(f"--country-map: {path} does not exist")
        print(f"note: no country map ({path}): the judges will see no country")
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["publisher"]: r.get("country") or "" for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--ranked", default="bm25,pipeline", help="ranked runs judged to --depth and checked by the humans (WORK/runs/<tag>.trec)")
    ap.add_argument("--sets", default="", help="strict result sets, one per writer (WORK/runs/<tag>_sets.jsonl); none for the practice round")
    ap.add_argument("--ai-only", default="", help="ranked runs judged to --depth by the AI only: the generous version")
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--judges", default="R,M", help="the two human judges' ids")
    ap.add_argument("--check-queries", type=int, default=25)
    ap.add_argument("--check-seed", default="20261116")
    ap.add_argument("--set-sample", type=int, default=20)
    ap.add_argument("--shared", type=int, default=150)
    ap.add_argument("--shared-queries", type=int, default=15)
    ap.add_argument("--context-chars", type=int, default=500, help="characters of the explanation shown to the judges under the claim")
    ap.add_argument("--country-map", default=None, help="publisher -> country CSV (snapshot/publisher_map.py); default "
                    "WORK/stats/publishers_map.csv when it exists. The country is shown to the judges with the fact-checker")
    args = ap.parse_args()
    work = Path(args.work)
    ranked, sets_tags, ai_only, judges = names(args.ranked), names(args.sets), names(args.ai_only), names(args.judges)
    if len(judges) != 2:
        raise SystemExit("--judges must name the two human judges")
    queries = load_queries(args.queries)
    qids = [q["query_id"] for q in queries]
    corpus = {d["cr_iri"]: d for d in kg.load_corpus(work / "corpus.jsonl")}
    country = load_country_map(Path(args.country_map) if args.country_map else work / "stats" / "publishers_map.csv",
                               required=bool(args.country_map))
    runs = {s: read_run(work / "runs" / f"{s}.trec") for s in ranked + ai_only}
    sets = {}
    for s in sets_tags:
        p = work / "runs" / f"{s}_sets.jsonl"
        if not p.exists():
            raise SystemExit(f"missing: {p}")
        sets[s] = {r["query_id"]: r["doc_ids"] for r in read_jsonl(p)}

    # ---- the pool ----
    pool_ranks: dict[str, dict[str, dict[str, int]]] = {}
    per_query = []
    for qid in qids:
        ranks: dict[str, dict[str, int]] = {}
        lists = {s: [d for d, _ in runs[s].get(qid, [])] for s in ranked + ai_only}
        for s in sets_tags:
            for r, d in enumerate(sets[s].get(qid, []), 1):
                ranks.setdefault(d, {})[s] = r
        big_b = max([len(sets[s].get(qid, [])) for s in sets_tags] or [0])
        for s in ranked:
            down_to = max(args.depth, big_b)
            for r, d in enumerate(lists[s][:down_to], 1):
                ranks.setdefault(d, {})[s] = r
        for s in ai_only:
            for r, d in enumerate(lists[s][:args.depth], 1):
                ranks.setdefault(d, {})[s] = r
        pool_ranks[qid] = ranks
        per_query.append({"query_id": qid, "pairs": len(ranks), "largest_set": big_b,
                          "beyond_depth_for_B": sum(1 for d, rs in ranks.items()
                                                    if all(s in ranked and r > args.depth for s, r in rs.items()))})

    # ---- the check sample ----
    # The seed is used as a number when it is one: random.Random(20261116) and random.Random("20261116") draw different
    # queries; the number is the documented seed.
    seed = int(args.check_seed) if str(args.check_seed).isdigit() else args.check_seed
    draw = random.Random(seed).sample(sorted(qids), min(args.check_queries, len(qids)))
    check = {"seed": args.check_seed, "depth": args.depth, "set_sample": args.set_sample, "ranked": ranked, "sets": sets_tags,
             "judges": judges, "queries": draw, "per_query": {}}
    for qid in draw:
        docs = set()
        for s in ranked:
            docs |= {d for d, rs in pool_ranks[qid].items() if rs.get(s, 10**9) <= args.depth}
        samples = {}
        for s in sets_tags:
            members = list(sets[s].get(qid, []))
            pick = members if len(members) <= args.set_sample else random.Random(f"{args.check_seed}:{s}:{qid}").sample(members, args.set_sample)
            samples[s] = sorted(pick)
            docs |= set(pick)
        check["per_query"][qid] = {"docs": sorted(docs), "set_samples": samples}
    # the shared documents: one at a time from each of the first --shared-queries check queries, in turn
    rng = random.Random(f"{args.check_seed}:shared")
    first = draw[:args.shared_queries]
    queues = {q: rng.sample(check["per_query"][q]["docs"], len(check["per_query"][q]["docs"])) for q in first}
    shared = []
    while len(shared) < args.shared and any(queues.values()):
        for q in first:
            if queues[q] and len(shared) < args.shared:
                shared.append((q, queues[q].pop()))
    shared_by_q: dict[str, set] = {}
    for q, d in shared:
        shared_by_q.setdefault(q, set()).add(d)
    # the owner of each check query: the judge with fewer documents so far
    load = {j: 0 for j in judges}
    assignments = {j: [] for j in judges}
    for qid in draw:
        docs = check["per_query"][qid]["docs"]
        sh = shared_by_q.get(qid, set())
        owner = min(judges, key=lambda j: (load[j], judges.index(j)))
        other = judges[1 - judges.index(owner)]
        check["per_query"][qid]["owner"] = owner
        check["per_query"][qid]["shared"] = sorted(sh)
        assignments[owner] += [[qid, d] for d in docs]
        assignments[other] += [[qid, d] for d in sorted(sh)]
        load[owner] += len(docs)
        load[other] += len(sh)
    in_check = {(q, d) for q in draw for d in check["per_query"][q]["docs"]}

    # ---- write ----
    pool_rows = []
    for qid in qids:
        for d, rs in sorted(pool_ranks[qid].items()):
            doc = corpus.get(d, {})
            pool_rows.append({"query_id": qid, "doc_id": d, "ranks": rs, "min_rank": min(rs.values()), "check": (qid, d) in in_check,
                              "claim_text": doc.get("claim_text", ""), "article_start": (doc.get("article_text") or "")[:args.context_chars],
                              "date_published": doc.get("date_published"), "author_name": doc.get("author_name"),
                              "country": country.get(doc.get("author_name") or "", ""), "rating": doc.get("rating")})
    out = work / "pool"
    write_jsonl(out / "pool.jsonl", pool_rows)
    write_json(out / "check.json", check)
    write_json(out / "shared.json", [list(x) for x in shared])
    write_json(out / "assignments.json", assignments)
    n_check = len(in_check)
    beyond = sum(r["beyond_depth_for_B"] for r in per_query)
    lines = ["# Pool summary", "",
             f"ranked systems (to depth {args.depth}, and to min(B, run length) for a set larger than {args.depth}): {', '.join(ranked)}; "
             f"strict sets (whole): {', '.join(sets_tags)}; AI-only (to depth {args.depth}): {', '.join(ai_only) or 'none'}",
             f"pairs for the AI judge: {len(pool_rows)} over {len(qids)} queries ({beyond} of them only because a set is larger than {args.depth})",
             f"check sample: {len(draw)} queries (seed {args.check_seed}), {n_check} documents, {len(shared)} judged by both; "
             + ", ".join(f"{j}: {len(assignments[j])} documents" for j in judges),
             f"check queries in order: {', '.join(draw)}", "",
             "| query | pairs | largest strict set | pairs only for B | in check | owner | shared |", "|---|---:|---:|---:|---|---|---:|"]
    for r in per_query:
        pq = check["per_query"].get(r["query_id"])
        lines.append(f"| {r['query_id']} | {r['pairs']} | {r['largest_set']} | {r['beyond_depth_for_B']} | "
                     f"{'yes (' + str(draw.index(r['query_id']) + 1) + ')' if pq else ''} | {pq['owner'] if pq else ''} | {len(pq['shared']) if pq else ''} |")
    (out / "pool_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:6]))
    print(f"wrote {out / 'pool.jsonl'}, check.json, shared.json, assignments.json, pool_summary.md")
    if len(shared) < args.shared:
        print(f"note: only {len(shared)} shared documents could be placed (the first {args.shared_queries} check queries have fewer documents)")


if __name__ == "__main__":
    main()
