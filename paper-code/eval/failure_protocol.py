"""
failure_protocol.py -- why the graph misses relevant fact-checks, and what it
returns that is not relevant (fixed protocol, no judgement
calls in the code).

For every RELEVANT document (qrels) that is NOT in the graph set of its query,
exactly one box, decided in this order:
  NO_ENTITIES        the document has no entity at all on the node searched (--node; the paper uses cr,
                     the explanation node, because every claim-level entity is Mars); split into its three
                     kinds (the same rule as stats/plot_archive.py): no_article (the
                     document has no explanation text), after_link_stop (an explanation, published from the month
                     entity linking stopped, --link-stop, on), never_linked (an explanation, older or undated,
                     and still no entity);
  ENTITY_MISMATCH    it has entities, but none of the formula's entities is among them;
  PARTIAL_MATCH      it mentions at least one formula entity but does not satisfy
                     the formula. If the formula's author wrote down concepts
                     they could not express as entities (not_expressible), the
                     row is flagged 'author_flagged_not_expressible' = yes; the
                     paper's box NOT_EXPRESSIBLE is PARTIAL_MATCH with that flag.
For every document IN the set that is judged NOT relevant: FALSE_HIT.
Set members without a label are counted as UNJUDGED_IN_SET.

Inputs: WORK/runs/<graph-tag>_sets.jsonl, WORK/entities_<node>.jsonl (entities_cr.jsonl in the paper), the qrels,
the query file, WORK/corpus.jsonl.
Outputs: WORK/eval/failures_<name>.csv (one row per miss or false hit),
         failures_<name>.md (shares per box, 5 examples per box),
         failures_<name>_sample.csv (a random sample of 30 mismatches and 30 false hits for manual sub-typing).

Run it once per writer (--graph-tag graph_annotator_a_cr, then graph_annotator_b_cr, each with its own --name).

Usage: python eval/failure_protocol.py --work WORK --queries Q.json --qrels WORK/qrels/qrels_ai.txt --graph-tag graph_annotator_a_cr --node cr --name ai_a
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import kg  # noqa: E402
from common.jsonl import read_jsonl  # noqa: E402
from common.queries import load_queries, pick_formula, term_uri_map  # noqa: E402
from common.trec import read_qrels  # noqa: E402

BOXES = ["NO_ENTITIES", "ENTITY_MISMATCH", "PARTIAL_MATCH", "FALSE_HIT", "UNJUDGED_IN_SET"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--graph-tag", default="graph")
    ap.add_argument("--node", default="cr", choices=kg.NODE_CHOICES, help="which entity index the graph run used (cr, the explanations, in the paper)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--rng-seed", type=int, default=42)
    ap.add_argument("--link-stop", default="2025-04", help="the month entity linking of the explanations stopped (stats/plot_archive.py)")
    args = ap.parse_args()
    work = Path(args.work)
    rng = random.Random(args.rng_seed)
    qrels = read_qrels(args.qrels)
    queries = {q["query_id"]: q for q in load_queries(args.queries)}
    sets = {r["query_id"]: r for r in read_jsonl(work / "runs" / f"{args.graph_tag}_sets.jsonl")}
    ents = kg.load_entity_index(work / f"entities_{args.node}.jsonl")
    corpus = {d["cr_iri"]: d for d in kg.load_corpus(work / "corpus.jsonl")}
    texts = {k: d["claim_text"] for k, d in corpus.items()}

    def no_entities_kind(d: str) -> str:
        doc = corpus.get(d, {})
        if not (doc.get("article_text") or "").strip():
            return "no_article"
        if (doc.get("date_published") or "")[:7] >= args.link_stop:
            return "after_link_stop"
        return "never_linked"
    rows = []
    for qid, s in sets.items():
        q = queries.get(qid)
        f = pick_formula(q, s.get("author_id")) if q else None
        if not f or not f.get("formula"):
            continue
        uris = set(term_uri_map(f).values())
        flagged = bool(f.get("not_expressible"))
        in_set = set(s["doc_ids"])
        rel = {d for d, r in qrels.get(qid, {}).items() if r == 1}
        for d in sorted(rel - in_set):
            de = ents.get(d, set())
            if not de:
                box = "NO_ENTITIES"
            elif not (de & uris):
                box = "ENTITY_MISMATCH"
            else:
                box = "PARTIAL_MATCH"
            rows.append({"query_id": qid, "doc_id": d, "box": box, "no_entities_kind": no_entities_kind(d) if box == "NO_ENTITIES" else "",
                         "n_doc_entities": len(de),
                         "formula_entities_present": ";".join(sorted(de & uris)),
                         "author_flagged_not_expressible": "yes" if flagged else "no",
                         "doc_entities": ";".join(sorted(de))[:500], "claim_text": texts.get(d, "")[:300]})
        for d in sorted(in_set):
            lab = qrels.get(qid, {}).get(d)
            if lab == 0:
                rows.append({"query_id": qid, "doc_id": d, "box": "FALSE_HIT", "no_entities_kind": "", "n_doc_entities": len(ents.get(d, set())),
                             "formula_entities_present": ";".join(sorted(ents.get(d, set()) & uris)), "author_flagged_not_expressible": "yes" if flagged else "no",
                             "doc_entities": ";".join(sorted(ents.get(d, set())))[:500], "claim_text": texts.get(d, "")[:300]})
            elif lab is None:
                rows.append({"query_id": qid, "doc_id": d, "box": "UNJUDGED_IN_SET", "no_entities_kind": "", "n_doc_entities": len(ents.get(d, set())),
                             "formula_entities_present": "", "author_flagged_not_expressible": "", "doc_entities": "", "claim_text": texts.get(d, "")[:300]})
    out = work / "eval"
    out.mkdir(parents=True, exist_ok=True)
    fields = ["query_id", "doc_id", "box", "no_entities_kind", "n_doc_entities", "formula_entities_present", "author_flagged_not_expressible", "doc_entities", "claim_text"]
    with open(out / f"failures_{args.name}.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    c = Counter(r["box"] for r in rows)
    misses = [r for r in rows if r["box"] in ("NO_ENTITIES", "ENTITY_MISMATCH", "PARTIAL_MATCH")]
    n_miss = len(misses)
    n_flag = sum(1 for r in misses if r["box"] == "PARTIAL_MATCH" and r["author_flagged_not_expressible"] == "yes")
    kinds = Counter(r["no_entities_kind"] for r in rows if r["box"] == "NO_ENTITIES")
    md = [f"# Failure protocol ({args.name} qrels)", "",
          f"relevant documents missed by the graph set: {n_miss}; of which NO_ENTITIES {c['NO_ENTITIES']} ({100*c['NO_ENTITIES']/n_miss:.0f}%), "
          f"ENTITY_MISMATCH {c['ENTITY_MISMATCH']} ({100*c['ENTITY_MISMATCH']/n_miss:.0f}%), PARTIAL_MATCH {c['PARTIAL_MATCH']} ({100*c['PARTIAL_MATCH']/n_miss:.0f}%), "
          f"of which with a not-expressible note by the author (the paper's NOT_EXPRESSIBLE box): {n_flag}" if n_miss else "no missed relevant documents",
          "NO_ENTITIES by kind: " + ", ".join(f"{k} {kinds[k]}" for k in ("no_article", "after_link_stop", "never_linked"))
          + f" (linking stopped in {args.link_stop})",
          f"set members judged not relevant (FALSE_HIT): {c['FALSE_HIT']}; set members without a label: {c['UNJUDGED_IN_SET']}", ""]
    by_box = defaultdict(list)
    for r in rows:
        by_box[r["box"]].append(r)
    for box in BOXES:
        ex = by_box.get(box, [])
        if not ex:
            continue
        md += [f"## {box}: {len(ex)} rows; examples", ""]
        for r in rng.sample(ex, min(5, len(ex))):
            md.append(f"- [{r['query_id']}] {r['claim_text'][:160]} (entities: {r['doc_entities'][:120] or 'none'})")
        md.append("")
    (out / f"failures_{args.name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sample = rng.sample(by_box["ENTITY_MISMATCH"], min(30, len(by_box["ENTITY_MISMATCH"]))) + rng.sample(by_box["FALSE_HIT"], min(30, len(by_box["FALSE_HIT"])))
    with open(out / f"failures_{args.name}_sample.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + ["manual_subtype"])
        w.writeheader()
        for r in sample:
            w.writerow({**r, "manual_subtype": ""})
    print("\n".join(md[:4]))
    print(f"wrote {out / ('failures_' + args.name + '.csv')} ({len(rows)} rows), .md, _sample.csv ({len(sample)} rows)")


if __name__ == "__main__":
    main()
