"""
ranked_entities.py -- the generous version of a writer's formula: coordination-level ranking over the formula's
CONCEPTS. A concept is one part joined by AND at the top of the formula:
("Ukraine" OR "Kyiv") AND "Biological agent" has two. Every document is scored
by the number of concepts it satisfies; documents that satisfy none are not
returned; documents with the same number are ordered newest first, as the ClaimsKG
Explorer lists a result, then by IRI; the score written falls strictly with the
position, so evaluation tools keep this order. Output: the top --depth.
A strict formula returns only the documents that satisfy every concept, so it
is empty as soon as one concept matches nothing; the generous version ranks the
documents that satisfy all concepts first, then all but one, and so on. A part
under NOT is a reason to exclude, never a concept to match, so it is not counted.
Judged by the AI judge only.

Usage: python systems/ranked_entities.py --queries Q.json --work WORK --node cr --author annotator_a --tag relaxed_annotator_a [--depth 100]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import kg  # noqa: E402
from common.boolean import And, Not, Or, evaluate, parse, positive_terms, terms  # noqa: E402
from common.queries import load_queries, pick_formula, term_uri_map  # noqa: E402
from common.trec import write_run  # noqa: E402
from systems.graph_boolean import sortable_date  # noqa: E402


def concepts(node) -> list:
    """The parts joined by AND at the top of the formula; a nested AND counts its own parts, so
    (a AND b) AND c has three; parts under NOT are left out. A formula without AND is one concept."""
    if isinstance(node, Not):
        return []
    if isinstance(node, And):
        out = []
        for c in node.children:
            out += concepts(c) if isinstance(c, And) else ([] if isinstance(c, Not) else [c])
        return out
    return [node]


def has_not(node) -> bool:
    if isinstance(node, Not):
        return True
    return isinstance(node, (And, Or)) and any(has_not(c) for c in node.children)


def concept_docs(concept, term_sets: dict[str, set[str]], universe: set[str]) -> set[str]:
    """The documents that satisfy one concept. A concept with a NOT inside (rare) is read by its
    positive terms only, so that a NOT can never make a concept match almost the whole archive."""
    if has_not(concept):
        return set().union(*(term_sets.get(t, set()) for t in positive_terms(concept)))
    return evaluate(concept, term_sets, universe)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--store", default=None, help="needed only if WORK/entities_<node>.jsonl does not exist yet")
    ap.add_argument("--node", default="cr", choices=kg.NODE_CHOICES)
    ap.add_argument("--author", default=None)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--tag", default="entities")
    args = ap.parse_args()
    work = Path(args.work)
    corpus = kg.load_corpus(work / "corpus.jsonl")
    dates = {d["cr_iri"]: d.get("date_published") for d in corpus}
    universe = set(dates)
    idx_path = work / f"entities_{args.node}.jsonl"
    if not idx_path.exists():
        if not args.store:
            raise SystemExit(f"{idx_path} missing; pass --store to build it")
        kg.build_entity_index(kg.open_store(args.store), args.node, idx_path)
    inv = kg.invert_entity_index(kg.load_entity_index(idx_path))
    queries = load_queries(args.queries)
    runs = {}
    for q in queries:
        f = pick_formula(q, args.author)
        if not f or not f.get("formula"):
            runs[q["query_id"]] = []
            continue
        uris = term_uri_map(f)
        node = parse(f["formula"])
        term_sets = {t: inv.get(uris.get(t, ""), set()) & universe for t in terms(node)}
        counts: dict[str, int] = {}
        for concept in concepts(node):
            for d in concept_docs(concept, term_sets, universe):
                counts[d] = counts.get(d, 0) + 1
        # More concepts first; among documents with the same number of concepts, newest first, as the
        # ClaimsKG Explorer lists a search result; same date: by IRI. The score
        # written falls strictly with the position, so that no evaluation tool can reorder the ties by
        # document id (trec_eval would); Anserini does the same for its own tie order (Lin and Yang 2019).
        ordered = sorted(counts, key=lambda d: (-counts[d], -int(sortable_date(dates.get(d)).replace("-", "") or 0), d))
        top = ordered[:args.depth]
        runs[q["query_id"]] = [(d, float(len(top) - i)) for i, d in enumerate(top)]
        print(f"  {q['query_id']} [{f.get('author_id')}] scored docs={len(counts)} max score={max(counts.values()) if counts else 0}")
    write_run(work / "runs" / f"{args.tag}.trec", runs, args.tag)
    print(f"wrote {work / 'runs' / (args.tag + '.trec')}")


if __name__ == "__main__":
    main()
