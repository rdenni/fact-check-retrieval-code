"""
fake_labels.py -- pretend human judges and pretend AI judges for the toy run.

Writes WORK/labels/labels_<judge>.jsonl for the documents assigned to each human judge (WORK/pool/assignments.json),
from tests/toy/truth.json with some random errors, and WORK/judge/labels_llm_<name>.jsonl with a label for EVERY pair
of the pool, one file per pretend AI judge (three by default, so that the majority candidate is built too), so that
adjudicate.py, validate_judge.py, build_qrels.py and eval/*.py can be exercised without any model. NEVER use on real data.
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonl import read_jsonl, read_json, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--annotators", default="R,M")
    ap.add_argument("--error-rate", type=float, default=0.1)
    ap.add_argument("--llm-names", default="gptoss120b,llama33_70b,deepseekv32")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    work = Path(args.work)
    truth = {q: set(v) for q, v in read_json(args.truth).items()}
    pool = list(read_jsonl(work / "pool" / "pool.jsonl"))
    assignments = read_json(work / "pool" / "assignments.json")
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for a in args.annotators.split(","):
        rows = []
        for qid, doc in assignments[a]:
            lab = "y" if doc in truth.get(qid, set()) else "n"
            if rng.random() < args.error_rate:
                lab = "n" if lab == "y" else "y"
            rows.append({"annotator": a, "query_id": qid, "doc_id": doc, "label": lab, "seconds": round(rng.uniform(3, 30), 1),
                         "ts": ts, "kind": "pool", "note": None, "guideline_sha256": "toy"})
        write_jsonl(work / "labels" / f"labels_{a}.jsonl", rows)
    for i, name in enumerate(args.llm_names.split(",")):
        err = args.error_rate * (1 + i) / 2  # the first model is the best
        rows = []
        for r in pool:
            lab = "y" if r["doc_id"] in truth.get(r["query_id"], set()) else "n"
            if rng.random() < err:
                lab = "n" if lab == "y" else "y"
            rows.append({"query_id": r["query_id"], "doc_id": r["doc_id"], "label": lab, "model": f"fake/{name}", "precision": "none",
                         "base_url": "none", "prompt_sha256": "toy", "seconds": 0.1, "raw": "YES" if lab == "y" else "NO", "ts": ts})
        write_jsonl(work / "judge" / f"labels_llm_{name}.jsonl", rows)
    print("fake labels written")


if __name__ == "__main__":
    main()
