"""
trec.py -- read and write the two standard TREC text formats.

Run file (one line per retrieved document):
    <query_id> Q0 <doc_id> <rank> <score> <tag>
Qrels file (one line per judged document):
    <query_id> 0 <doc_id> <relevance>

Both are plain text, space separated, UTF-8. doc_id is the CimpleKG
ClaimReview IRI (no spaces, so it is safe as a token).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def write_run(path: str | Path, ranked: dict[str, list[tuple[str, float]]], tag: str) -> None:
    """ranked: query_id -> list of (doc_id, score), best first.

    Scores must be non-increasing down the list; we write the given order and
    the given score, and never re-sort, so a caller who wants ties to be
    visible can pass equal scores.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid in ranked:
            for rank, (doc_id, score) in enumerate(ranked[qid], 1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {tag}\n")


def read_run(path: str | Path) -> dict[str, list[tuple[str, float]]]:
    """Returns query_id -> list of (doc_id, score) in trec_eval's order: score descending,
    and documents with the same score by doc_id in descending lexicographic order.

    This is the order trec_eval itself evaluates: it ignores the rank column and the line
    order ("internally ranks are assigned by sorting by the sim field with ties broken
    deterministicly (using docno)", trec_eval README; form_res_rels.c, comp_sim_docno:
    sim descending, then strcmp(docno2, docno1)), and ir_measures/pytrec_eval do the same
    Reading every run this way makes the pool, the oracle, the failure
    analysis and the measures see one and the same ranking. A run whose scores fall strictly down the file (the
    graph sets, the generous version) keeps its file order, because nothing is tied.
    """
    out: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, doc_id, _rank, score, _tag = parts[:6]
            out[qid].append((doc_id, float(score)))
    for docs in out.values():
        docs.sort(key=lambda x: x[0], reverse=True)  # ties: doc_id descending ...
        docs.sort(key=lambda x: -x[1])               # ... kept by the stable sort on score descending
    return dict(out)


def write_qrels(path: str | Path, qrels: dict[str, dict[str, int]]) -> None:
    """qrels: query_id -> {doc_id: relevance (0 or 1)}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(qrels):
            for doc_id in sorted(qrels[qid]):
                f.write(f"{qid} 0 {doc_id} {int(qrels[qid][doc_id])}\n")


def read_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, doc_id, rel = parts[:4]
            out[qid][doc_id] = int(rel)
    return dict(out)
