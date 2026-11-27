"""
pack_for_judge.py -- put what the AI judge needs, and nothing else, into one folder (and a zip), which
judge/run_judges.py reads (through an API, judge/RUN-WITH-API.md, or on a GPU machine). Data never goes through
GitHub; if the folder must travel, it is sent by a drive link.

The AI judge labels every pooled pair zero-shot (judge/assess.py), so the bundle needs no human label: it can be
sent as soon as the pool is built, while the human judges label the check sample.

Written into the bundle, slimmed to the fields the prompt uses (judge/prompts.py):
  pool/pool.jsonl   one line per pair: query_id, doc_id, claim_text, article_start, date_published, author_name, country
                    (not the verdict, not which system returned the document, not whether it is in the check sample);
  queries.json      one entry per query: query_id, free_text, background (not the formulas, not the provenance);
  MANIFEST.json     the counts, the prompt version, the creation time and the sha256 of every file
                    (judge/run_judges.py reads the query file name and the number of pairs from it).
Refused: a bundle folder that exists and is not empty.

Usage: python judge/pack_for_judge.py --work WORK --queries Q.json --out BUNDLE [--zip]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonl import read_jsonl, write_jsonl, write_json  # noqa: E402
from common.queries import load_queries  # noqa: E402
from judge import prompts  # noqa: E402

POOL_FIELDS = ("query_id", "doc_id", "claim_text", "article_start", "date_published", "author_name", "country")
QUERY_FIELDS = ("query_id", "free_text", "background")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True, help="the bundle folder to create (must not exist or be empty)")
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip")
    args = ap.parse_args()
    work, out = Path(args.work), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; choose a new folder")
    pool_path = work / "pool" / "pool.jsonl"
    if not pool_path.exists():
        raise SystemExit(f"missing: {pool_path} (run pool/build_pool.py first)")
    pool = [{k: r.get(k) for k in POOL_FIELDS} for r in read_jsonl(pool_path)]
    qids = {r["query_id"] for r in pool}
    queries = [{k: q.get(k) for k in QUERY_FIELDS} for q in load_queries(args.queries) if q["query_id"] in qids]
    missing = sorted(qids - {q["query_id"] for q in queries})
    if missing:
        raise SystemExit(f"the pool has queries that {args.queries} lacks: {missing[:5]}")
    write_jsonl(out / "pool" / "pool.jsonl", pool)
    write_json(out / "queries.json", queries)
    files = ["pool/pool.jsonl", "queries.json"]
    counts = {"pairs": len(pool), "queries": len(qids)}
    manifest = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "queries": "queries.json",
                "prompt_sha256": prompts.prompt_fingerprint(),
                "counts": counts, "sha256": {rel: sha256(out / rel) for rel in files}}
    write_json(out / "MANIFEST.json", manifest)
    print(f"bundle {out}: {counts['pairs']} pairs over {counts['queries']} queries; prompt sha256 {prompts.prompt_fingerprint()[:12]}")
    if args.zip:
        z = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"zip: {z} (sha256 {sha256(Path(z))})")


if __name__ == "__main__":
    main()
