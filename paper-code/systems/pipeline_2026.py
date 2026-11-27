"""
pipeline_2026.py -- system 3: BM25 + dense retriever -> RRF -> reranker.

Three steps, run as three processes so that the embedder and the reranker
are never in memory together:

  embed     load the embedder; encode every document text and every query; save.
  retrieve  no model: BM25 top-100 (systems/bm25.py) and dense top-100
            (exact search over the saved vectors); fuse with reciprocal rank
            fusion, k = 60 (Cormack et al. 2009): score(d) = sum over the two
            lists of 1 / (60 + rank); save the union as candidates.
  rerank    load the reranker; score every (query, candidate) pair; keep the
            top-100 by P(yes).

Runs written to WORK/runs/ (TREC format); the paper reports bm25.trec and pipeline.trec, the other two are the
first stages alone:
  bm25.trec       BM25 alone (also system 2 of the main table)
  dense.trec      the embedder alone
  fused.trec      BM25 + dense fused with RRF, no reranker
  pipeline.trec   the full pipeline (system 3 of the main table)

Usage:
  python systems/pipeline_2026.py embed    --work WORK --queries Q.json --profile laptop|cpu-test
  python systems/pipeline_2026.py retrieve --work WORK --queries Q.json --profile ... [--depth 100] [--rrf-k 60]
  python systems/pipeline_2026.py rerank   --work WORK --queries Q.json --profile ... [--out-depth 100]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import kg  # noqa: E402
from common.jsonl import write_json, read_json  # noqa: E402
from common.queries import load_queries  # noqa: E402
from common.trec import write_run  # noqa: E402
from profiles import get_profile  # noqa: E402

RRF_K = 60
CHANNEL_DEPTH = 100
OUTPUT_DEPTH = 100


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion by document id. rankings: lists of doc ids, best
    first (rank 1 = position 0). Returns (doc_id, score) sorted best first,
    ties broken by doc id so the order is reproducible."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for pos, d in enumerate(ranking):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + pos + 1)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def dense_dir(work: Path, profile: dict) -> Path:
    return work / f"dense_{profile['name']}"


def cmd_embed(args):
    from systems.dense import embed_corpus_and_queries
    profile = get_profile(args.profile, device=args.device, dtype=args.dtype, attn=args.attn,
                          embedder_revision=args.embedder_revision, text_mode=args.text_mode)
    work = Path(args.work)
    corpus = kg.load_corpus(work / "corpus.jsonl")
    queries = load_queries(args.queries)
    if args.limit:
        corpus = corpus[:args.limit]
    meta = embed_corpus_and_queries(profile, corpus, queries, dense_dir(work, profile), args.batch_size)
    print(json.dumps(meta, indent=2))


def cmd_retrieve(args):
    from systems.bm25 import Bm25Index
    from systems.dense import load_embeddings, exact_search
    profile = get_profile(args.profile)
    work = Path(args.work)
    queries = load_queries(args.queries)
    qids = [q["query_id"] for q in queries]
    # BM25 channel (identical to system 2).
    index = Bm25Index.load(work / "bm25_index")
    bm25_mode = json.loads((work / "bm25_index" / "params.json").read_text(encoding="utf-8")).get("text_mode", "claim")
    dense_mode = json.loads((dense_dir(work, profile) / "meta.json").read_text(encoding="utf-8")).get("text_mode", "claim")
    if bm25_mode != dense_mode:
        raise SystemExit(f"the BM25 index was built with text mode {bm25_mode!r} and the embeddings with {dense_mode!r}; rebuild one of them")
    bm25_hits = index.search([q["free_text"] for q in queries], args.depth)
    bm25_runs = dict(zip(qids, bm25_hits))
    # Dense channel.
    doc_emb, doc_ids, q_emb, q_ids = load_embeddings(dense_dir(work, profile))
    if q_ids != qids:
        raise SystemExit("query ids in the embedding folder differ from the query file; rerun 'embed'")
    dense_hits = exact_search(q_emb, doc_emb, doc_ids, args.depth)
    dense_runs = dict(zip(qids, dense_hits))
    # Fusion.
    fused_runs = {}
    candidates = {}
    for qid in qids:
        fused = rrf([[d for d, _ in bm25_runs[qid]], [d for d, _ in dense_runs[qid]]], args.rrf_k)
        fused_runs[qid] = fused
        candidates[qid] = [d for d, _ in fused]
    out = work / "runs"
    write_run(out / "bm25.trec", bm25_runs, "bm25")
    write_run(out / "dense.trec", dense_runs, "dense")
    write_run(out / "fused.trec", fused_runs, "fused")
    write_json(work / f"candidates_{profile['name']}.json", candidates)
    write_json(work / f"retrieve_{profile['name']}_meta.json",
               {"channel_depth": args.depth, "rrf_k": args.rrf_k, "dense_dir": str(dense_dir(work, profile)),
                "n_candidates": {q: len(c) for q, c in candidates.items()}})
    print(f"wrote bm25.trec, dense.trec, fused.trec; candidates per query: "
          f"min {min(len(c) for c in candidates.values())}, max {max(len(c) for c in candidates.values())}")


def cmd_rerank(args):
    from systems.rerank import rerank_candidates
    profile = get_profile(args.profile, device=args.device, dtype=args.dtype, attn=args.attn,
                          reranker_revision=args.reranker_revision, text_mode=args.text_mode)
    work = Path(args.work)
    queries = load_queries(args.queries)
    corpus = kg.load_corpus(work / "corpus.jsonl")
    texts = {d["cr_iri"]: kg.doc_text(d, args.text_mode) for d in corpus}
    candidates = read_json(work / f"candidates_{profile['name']}.json")
    runs, meta = rerank_candidates(profile, queries, candidates, texts, args.batch_size)
    runs = {q: r[:args.out_depth] for q, r in runs.items()}
    write_run(work / "runs" / "pipeline.trec", runs, "pipeline")
    write_json(work / f"rerank_{profile['name']}_meta.json", meta)
    print(json.dumps(meta, indent=2))
    print(f"wrote {work / 'runs' / 'pipeline.trec'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("embed", "retrieve", "rerank"):
        s = sub.add_parser(name)
        s.add_argument("--work", required=True)
        s.add_argument("--queries", required=True)
        s.add_argument("--profile", required=True)
        if name in ("embed", "rerank"):
            s.add_argument("--text-mode", default="article", choices=kg.TEXT_MODES, help="which text the model reads per document (the paper: explanation)")
            s.add_argument("--device", default=None)
            s.add_argument("--dtype", default=None, choices=(None, "float32", "float16", "bfloat16"))
            s.add_argument("--attn", default=None, help="sdpa (default) or flash_attention_2 (Linux with flash-attn)")
            s.add_argument("--batch-size", type=int, default=None)
        if name == "embed":
            s.add_argument("--embedder-revision", default=None)
            s.add_argument("--limit", type=int, default=None, help="encode only the first N documents (smoke test)")
        if name == "retrieve":
            s.add_argument("--depth", type=int, default=CHANNEL_DEPTH)
            s.add_argument("--rrf-k", type=int, default=RRF_K)
        if name == "rerank":
            s.add_argument("--reranker-revision", default=None)
            s.add_argument("--out-depth", type=int, default=OUTPUT_DEPTH)
    args = ap.parse_args()
    {"embed": cmd_embed, "retrieve": cmd_retrieve, "rerank": cmd_rerank}[args.cmd](args)


if __name__ == "__main__":
    main()
