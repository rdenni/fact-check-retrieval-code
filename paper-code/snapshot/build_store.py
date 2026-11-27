"""
build_store.py -- turn the mirrored N-Triples file into the Oxigraph store the
paper code reads (--store), applying the cleaning rules and writing a manifest.

Rules, in order:
  1. Language label: the mirror already holds only ClaimReviews with inLanguage
     "English" (snapshot/mirror_english.py); the script checks and reports it.
  2. Scraping failures:
     a ClaimReview whose schema:text (the explanation text on the ClaimReview node)
     is shared by at least --dup-threshold (5) ClaimReviews is a scraper error
     page reused across broken sources; it is removed with its Claim.
  3. Optional, --publisher-map CSV with columns publisher, ifcn_status:
     ClaimReviews whose publisher is not marked "verified" are removed. The
     archive is built from IFCN signatories at collection time (CimpleKG paper,
     step 3), so this only removes publishers whose signatory status has since
     expired, if the paper decides to be that strict. The CSV comes from
     snapshot/publisher_map.py plus hand checks.
  4. No claim text: a ClaimReview without a Claim (itemReviewed) carrying a
     non-empty text cannot be a document (common/kg.py reads only those), so
     it is removed here too and the manifest's count equals the corpus size.
  5. Language of the text (snapshot/language_rule.py; off
     with --no-language-rule): the inLanguage label is wrong for about a third
     of the "English" fact-checks (Hindi, Bengali, Italian, Spanish... see the
     rule's file), so every remaining ClaimReview's claim and explanation are
     checked, and the ones that are not English are removed with their Claim.
     Every decision is written to language_decisions.jsonl next to the manifest.

Steps: bulk-load mirror.nt into the new store; compute the removal sets with
SPARQL; delete every triple whose subject is a removed ClaimReview or its
Claim; write snapshot_manifest.json with the counts and the rules.

Usage:
  python snapshot/build_store.py --mirror SNAPSHOT_DIR/mirror.nt --store STORE_DIR [--publisher-map map.csv] [--dup-threshold 5] [--no-language-rule]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import kg  # noqa: E402
from common.jsonl import write_jsonl  # noqa: E402
from snapshot import language_rule  # noqa: E402


def subjects_quads(store, subject):
    from pyoxigraph import NamedNode
    return list(store.quads_for_pattern(NamedNode(subject), None, None, None))


def remove_claim_reviews(store, cr_iris) -> int:
    """Delete every triple whose subject is one of the ClaimReviews, and every triple of their Claims that no
    surviving ClaimReview still reviews (a Claim node is shared when the same claim was checked several times;
    deleting it with the first removed review would silently strip the text from the kept ones). Returns the count."""
    cr_iris = set(cr_iris)
    if not cr_iris:
        return 0
    reviewers: dict[str, set[str]] = {}
    for row in store.query(f'SELECT ?cr ?claim WHERE {{ ?cr a <{kg.C_CLAIMREVIEW}> ; <{kg.P_ITEM_REVIEWED}> ?claim }}'):
        reviewers.setdefault(row["claim"].value, set()).add(row["cr"].value)
    n = 0
    for cr in sorted(cr_iris):
        claims = [q.object.value for q in subjects_quads(store, cr) if q.predicate.value == kg.P_ITEM_REVIEWED]
        orphaned = [c for c in claims if not (reviewers.get(c, set()) - cr_iris)]
        for subj in [cr] + orphaned:
            for q in subjects_quads(store, subj):
                store.remove(q)
                n += 1
    store.flush()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mirror", required=True, help="mirror.nt from snapshot/mirror_english.py")
    ap.add_argument("--store", required=True, help="new Oxigraph store directory (created; refused if it exists unless --force)")
    ap.add_argument("--publisher-map", default=None)
    ap.add_argument("--dup-threshold", type=int, default=5)
    ap.add_argument("--no-language-rule", action="store_true", help="keep the non-English fact-checks (rule 5 off)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    from pyoxigraph import Store, RdfFormat
    store_dir = Path(args.store)
    if store_dir.exists():
        if not args.force:
            raise SystemExit(f"{store_dir} exists; pass --force to replace it")
        shutil.rmtree(store_dir)
    store_dir.mkdir(parents=True)
    store = Store(str(store_dir))
    print("loading the mirror (a few minutes for the whole English part)...", flush=True)
    store.bulk_load(path=str(args.mirror), format=RdfFormat.N_TRIPLES)
    n_triples = len(store)
    print(f"  {n_triples:,} triples", flush=True)

    # Rule 1: languages present (should be English only).
    langs = Counter()
    for row in store.query(f'SELECT ?l (COUNT(*) AS ?n) WHERE {{ ?cr a <{kg.C_CLAIMREVIEW}> ; <{kg.P_LANG}> ?l }} GROUP BY ?l'):
        langs[row["l"].value] = int(row["n"].value)
    n_cr = kg.count_crs(store)
    print(f"  ClaimReviews: {n_cr:,}; languages: {dict(langs)}", flush=True)

    # Rule 2: scraping failures.
    dup_texts = {row["t"].value for row in store.query(
        f'SELECT ?t (COUNT(*) AS ?n) WHERE {{ ?cr a <{kg.C_CLAIMREVIEW}> ; <{kg.P_TEXT}> ?t }} GROUP BY ?t HAVING (COUNT(*) >= {args.dup_threshold})')}
    failures = set()
    if dup_texts:
        for row in store.query(f'SELECT ?cr ?t WHERE {{ ?cr a <{kg.C_CLAIMREVIEW}> ; <{kg.P_TEXT}> ?t }}'):
            if row["t"].value in dup_texts:
                failures.add(row["cr"].value)
    print(f"  scraping-failure texts (shared by >= {args.dup_threshold}): {len(dup_texts)}; ClaimReviews removed for it: {len(failures):,}", flush=True)

    # Rule 3: publishers not verified by IFCN (optional).
    not_ifcn = set()
    if args.publisher_map:
        status = {}
        with open(args.publisher_map, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                status[row["publisher"]] = (row.get("ifcn_status") or "").strip().lower()
        for row in store.query(f'SELECT ?cr ?name WHERE {{ ?cr a <{kg.C_CLAIMREVIEW}> ; <{kg.P_AUTHOR}> ?o . ?o <{kg.P_NAME}> ?name }}'):
            if status.get(row["name"].value, "") != "verified":
                not_ifcn.add(row["cr"].value)
        print(f"  ClaimReviews whose publisher is not 'verified' in the map: {len(not_ifcn):,}", flush=True)

    n_removed_triples = remove_claim_reviews(store, failures | not_ifcn)

    # Rule 4: ClaimReviews without a claim text are not documents. Rule 5: the language of the text.
    print("  reading every remaining ClaimReview with its claim and explanation...", flush=True)
    records = list(kg.iter_claim_reviews(store))
    with_text = {r["cr_iri"] for r in records}
    all_crs = {row["cr"].value for row in store.query(f'SELECT ?cr WHERE {{ ?cr a <{kg.C_CLAIMREVIEW}> }}')}
    no_claim_text = all_crs - with_text
    print(f"  ClaimReviews without a claim text: {len(no_claim_text):,}", flush=True)
    n_removed_triples += remove_claim_reviews(store, no_claim_text)
    not_english = set()
    language_summary = None
    if not args.no_language_rule:
        decisions = language_rule.decide(records, progress=True)
        write_jsonl(store_dir.parent / "language_decisions.jsonl", decisions)
        language_summary = language_rule.summarise(decisions)
        not_english = {d["cr_iri"] for d in decisions if not d["english"]}
        print(f"  language rule {language_rule.VERSION}: not English {len(not_english):,} of {len(decisions):,}; by reason {language_summary['by_reason']}", flush=True)
        n_removed_triples += remove_claim_reviews(store, not_english)

    n_cr_after = kg.count_crs(store)
    sha = hashlib.sha256()
    with open(args.mirror, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    manifest = {"built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "mirror": str(args.mirror), "mirror_sha256": sha.hexdigest(),
                "triples_loaded": n_triples, "triples_after": len(store), "claimreviews_loaded": n_cr, "claimreviews_after": n_cr_after,
                "languages": dict(langs),
                "rules": {"language_label": "English (by the mirror)", "scraping_failures": f"ClaimReview text shared by >= {args.dup_threshold} ClaimReviews",
                          "ifcn": ("publisher must be 'verified' in " + args.publisher_map) if args.publisher_map else "not applied",
                          "no_claim_text": "ClaimReview without a Claim with a non-empty text",
                          "language_of_text": (f"snapshot/language_rule.py version {language_rule.VERSION}, decisions in language_decisions.jsonl") if not args.no_language_rule else "not applied"},
                "removed": {"scraping_failures": len(failures), "not_ifcn": len(not_ifcn), "no_claim_text": len(no_claim_text), "not_english": len(not_english),
                            "triples": n_removed_triples},
                "language_rule": language_summary}
    (store_dir.parent / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"store {store_dir}: {n_cr_after:,} ClaimReviews, {len(store):,} triples; manifest -> {store_dir.parent / 'snapshot_manifest.json'}")


if __name__ == "__main__":
    main()
