"""
graph_boolean.py -- system 1: the knowledge graph used as its papers use it.

What it does, per query:
  1. Takes the writer's Boolean formula over entities (common/boolean.py).
  2. For every entity term runs ONE SPARQL query on the store that returns the
     ClaimReviews whose explanation mentions that entity (--node cr, the
     ClaimReview node: the paper's runs); --node claim reads the Claim node,
     which the graph papers query in their examples but which holds a single
     entity on the English part of CimpleKG, and --node union reads both.
  3. Combines the sets with AND / OR / NOT.
  4. Orders the set by publication date, newest first (as CimpleKG's own
     example query does); missing dates last; ties broken by the IRI so the
     order is the same on every run.
  5. Records B = the size of the whole set, writes the whole set to
     graph_sets.jsonl, and writes the first --cap (default 1000) items as a
     TREC run whose score is (cap - rank + 1), so any tool that sorts by score
     keeps the date order.
No date, language, author or rating filter is applied by anything.

Backends: 'sparql' (one query per term on the store; the paper's wording) and
'index' (the same triples read once into memory from entities_<node>.jsonl,
built by run_all.py; faster for repeated runs and for the failure protocol).
tests/run_toy.py runs the index backend next to the SPARQL one; the two gave identical sets on the toy store.

Usage:
  python systems/graph_boolean.py --queries Q.json --store STORE --work WORK \
      [--node cr] [--author annotator_a] [--backend sparql] [--cap 1000]
Outputs in WORK/runs/: graph.trec, graph_sets.jsonl, graph_report.md
(with --author X and --tag T the file names carry the tag).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import kg  # noqa: E402
from common.boolean import drop_terms, parse, terms, evaluate, is_wide_not, to_string  # noqa: E402
from common.jsonl import write_jsonl  # noqa: E402
from common.queries import load_queries, pick_formula, term_uri_map  # noqa: E402
from common.trec import write_run  # noqa: E402


def sortable_date(date_str: str | None) -> str:
    """'yyyy-mm-dd' when the value starts with an ISO date, else '' (treated as undated)."""
    d = kg.date_key(date_str)
    if len(d) == 10 and d[:4].isdigit() and d[5:7].isdigit() and d[8:10].isdigit() and d[4] == "-" and d[7] == "-":
        return d
    return ""


def order_by_date(doc_ids: set[str], dates: dict[str, str | None]) -> list[str]:
    """Newest first; documents without a usable date last; inside a date, IRI order."""
    dated = [d for d in doc_ids if sortable_date(dates.get(d))]
    undated = sorted(d for d in doc_ids if not sortable_date(dates.get(d)))
    dated.sort(key=lambda d: (-int(sortable_date(dates.get(d)).replace("-", "")), d))
    return dated + undated


def run_graph(queries, term_lookup, universe: set[str], dates: dict[str, str | None],
              author: str | None, cap: int, allow_wide_not: bool, current_only: bool = False):
    """term_lookup(entity_uri) -> set of CR IRIs. Returns (runs, sets, report_rows)."""
    runs: dict[str, list[tuple[str, float]]] = {}
    sets = []
    report = []
    cache: dict[str, set[str]] = {}
    for q in queries:
        qid = q["query_id"]
        f = pick_formula(q, author)
        if f is None or not f.get("formula"):
            runs[qid] = []
            sets.append({"query_id": qid, "author_id": author, "formula": None, "B": 0, "doc_ids": [], "note": "no formula"})
            report.append((qid, author, "(none)", 0, 0, 0, 0.0))
            continue
        node = parse(f["formula"])
        formula_run = f["formula"]
        if current_only:
            # today's pages only: drop every other name of a page that the import added (as written, older names)
            node = drop_terms(node, {k for k, e in (f.get("entities") or {}).items() if e.get("alias_of")})
            if node is None:  # cannot happen with the import's formulas (today's page always stays); kept safe
                runs[qid] = []
                sets.append({"query_id": qid, "author_id": f.get("author_id"), "formula": None, "B": 0, "doc_ids": [], "note": "no term left"})
                report.append((qid, f.get("author_id"), "(no term left)", 0, 0, 0, 0.0))
                continue
            formula_run = to_string(node)
        if is_wide_not(node) and not allow_wide_not:
            raise SystemExit(f"{qid} [{f.get('author_id')}]: formula {f['formula']!r} is a bare negation; "
                             f"NOT must be inside an AND (A AND NOT B). Pass --allow-wide-not to run it anyway.")
        uris = term_uri_map(f)
        t0 = time.time()
        term_sets = {}
        for t in terms(node):
            uri = uris.get(t)
            if not uri:
                term_sets[t] = set()  # entity without a DBpedia URI can never match
                continue
            if uri not in cache:
                cache[uri] = term_lookup(uri)
            term_sets[t] = cache[uri]
        result = evaluate(node, term_sets, universe)
        ordered = order_by_date(result, dates)
        dt = time.time() - t0
        B = len(ordered)
        runs[qid] = [(d, float(cap - r + 1)) for r, d in enumerate(ordered[:cap], 1)]
        # Date ties around the cutoffs 10 and 20: how many documents share the date of the item at that rank.
        ties = {}
        for k in (10, 20):
            if B >= k:
                dk = sortable_date(dates.get(ordered[k - 1]))
                ties[k] = sum(1 for d in ordered if sortable_date(dates.get(d)) == dk) if dk else sum(1 for d in ordered if not sortable_date(dates.get(d)))
            else:
                ties[k] = 0
        n_undated = sum(1 for d in ordered if not sortable_date(dates.get(d)))
        sets.append({"query_id": qid, "author_id": f.get("author_id"), "formula": formula_run,
                     "terms": {t: uris.get(t) for t in terms(node)},
                     # terms that only repeat a page under the name the writer typed (import rule 'either page'):
                     # the oracle does not count them in the writer's budget of pages
                     "alias_terms": [t for t in terms(node) if ((f.get("entities") or {}).get(t) or {}).get("alias_of")],
                     "term_set_sizes": {t: len(s) for t, s in term_sets.items()},
                     "B": B, "n_undated": n_undated, "ties_at_10": ties[10], "ties_at_20": ties[20],
                     "doc_ids": ordered})
        report.append((qid, f.get("author_id"), formula_run, B, ties[10], ties[20], dt))
    return runs, sets, report


def write_report(path: Path, report, node: str, cap: int):
    lines = ["# Graph system report", "",
             f"Node queried: {node}. Set capped at {cap} in the TREC run; B is the size of the whole set.",
             "ties@k = number of documents that share the publication date of the document at rank k",
             "(with a set ordered newest first, everything inside a tie is in arbitrary IRI order).", "",
             "| query | author | formula | B | ties@10 | ties@20 | seconds |", "|---|---|---|---:|---:|---:|---:|"]
    for qid, author, formula, B, t10, t20, dt in report:
        lines.append(f"| {qid} | {author} | `{formula}` | {B} | {t10} | {t20} | {dt:.2f} |")
    Bs = [r[3] for r in report]
    if Bs:
        c = Counter("0" if b == 0 else "1-9" if b < 10 else "10-19" if b < 20 else "20-99" if b < 100 else "100-999" if b < 1000 else "1000+" for b in Bs)
        lines += ["", "B distribution: " + ", ".join(f"{k}: {c[k]}" for k in ["0", "1-9", "10-19", "20-99", "100-999", "1000+"] if c[k]),
                  f"median B = {sorted(Bs)[len(Bs)//2]}, max B = {max(Bs)}, queries with B = 0: {c['0']}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--store", required=True, help="Oxigraph store directory (read only)")
    ap.add_argument("--work", required=True, help="work directory; needs WORK/corpus.jsonl (run_all.py prepare)")
    ap.add_argument("--node", default="cr", choices=kg.NODE_CHOICES)
    ap.add_argument("--author", default=None, help="formula author to run (default: consensus, else the first)")
    ap.add_argument("--backend", default="sparql", choices=("sparql", "index"))
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--tag", default=None, help="file tag (default: graph, or graph_<author>_<node>)")
    ap.add_argument("--allow-wide-not", action="store_true")
    ap.add_argument("--current-names-only", action="store_true",
                    help="today's Wikipedia pages only, without the other names the import adds (default tag ends in _current); "
                         "eval/stale_names.py compares it with the normal run")
    args = ap.parse_args()

    work = Path(args.work)
    corpus = kg.load_corpus(work / "corpus.jsonl")
    universe = {d["cr_iri"] for d in corpus}
    dates = {d["cr_iri"]: d.get("date_published") for d in corpus}
    queries = load_queries(args.queries)
    tag = args.tag or ("graph" if not args.author and args.node == "claim" else f"graph_{args.author or 'default'}_{args.node}")
    if args.current_names_only and not args.tag:
        tag += "_current"

    if args.backend == "sparql":
        store = kg.open_store(args.store)

        def lookup(uri):
            return kg.crs_mentioning(store, uri, args.node) & universe
    else:
        idx_path = work / f"entities_{args.node}.jsonl"
        if not idx_path.exists():
            store = kg.open_store(args.store)
            kg.build_entity_index(store, args.node, idx_path)
        inv = kg.invert_entity_index(kg.load_entity_index(idx_path))

        def lookup(uri):
            return inv.get(uri, set()) & universe

    runs, sets, report = run_graph(queries, lookup, universe, dates, args.author, args.cap, args.allow_wide_not, args.current_names_only)
    out = work / "runs"
    out.mkdir(parents=True, exist_ok=True)
    write_run(out / f"{tag}.trec", runs, tag)
    write_jsonl(out / f"{tag}_sets.jsonl", sets)
    write_report(out / f"{tag}_report.md", report, args.node, args.cap)
    print(f"wrote {out / (tag + '.trec')}, {tag}_sets.jsonl, {tag}_report.md")
    for qid, author, formula, B, t10, t20, dt in report:
        print(f"  {qid} [{author}] B={B} ties@10={t10} ties@20={t20}  {formula}")


if __name__ == "__main__":
    main()
