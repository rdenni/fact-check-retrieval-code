"""
run_all.py -- run the three systems of the paper (and, optionally, the ranking over the entities of
one formula) on one query file, and write a manifest.

Steps (each can be run alone with --steps):
  prepare   corpus.jsonl (the document universe) and entities_claim.jsonl from the store
  graph     system 1  (systems/graph_boolean.py, SPARQL backend)
  bm25      system 2  (systems/bm25.py build + search)
  embed     system 3, step 1 (loads the embedder)
  retrieve  system 3, step 2 (no model; also writes dense.trec and fused.trec)
  rerank    system 3, step 3 (loads the reranker)
  appendix  the ranking over the entities of a formula (systems/ranked_entities.py; optional, --appendix)
  manifest  runs/manifest.json with every model, revision, parameter, file hash and timing

Profile (profiles.py):
  --profile laptop    the paper's runs: the 0.6B pair on an 8 GB laptop GPU
  --profile cpu-test  a tiny random model, to test the code only

Example (the paper's runs: the explanations, the explanation-level entities):
  python run_all.py --queries QUERIES.json --store STORE --work WORK --profile laptop --node cr --text-mode article
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import kg, textprep  # noqa: E402
from common.jsonl import write_json, read_json  # noqa: E402
from profiles import get_profile  # noqa: E402

HERE = Path(__file__).resolve().parent
ALL_STEPS = ["prepare", "graph", "bm25", "embed", "retrieve", "rerank", "appendix", "manifest"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(store_path: str, work: Path, node: str) -> tuple[int, int]:
    """Open the store in this function only, so its lock is released before the
    graph step opens it again in a child process."""
    import gc
    store = kg.open_store(store_path)
    n = kg.build_corpus_snapshot(store, work / "corpus.jsonl")
    m = kg.build_entity_index(store, node, work / f"entities_{node}.jsonl")
    del store
    gc.collect()
    return n, m


def run(cmd: list[str]) -> float:
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--store", default=None, help="Oxigraph store; needed for prepare and graph")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--steps", default=",".join(ALL_STEPS))
    ap.add_argument("--author", default=None, help="formula author for the graph system (default consensus/first)")
    ap.add_argument("--node", default="cr", choices=kg.NODE_CHOICES)
    ap.add_argument("--text-mode", default="article", choices=kg.TEXT_MODES, help="text the text systems read per document: the explanation (the paper's main runs), the claim sentence, or claim + explanation")
    ap.add_argument("--appendix", action="store_true", help="also run the ranking over the entities of the formula (systems/ranked_entities.py)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--attn", default=None)
    ap.add_argument("--snapshot-date", default="2026-10-02", help="date of the CimpleKG snapshot (metadata)")
    args = ap.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    timings = {}
    profile = get_profile(args.profile)

    if "prepare" in steps:
        if not args.store:
            raise SystemExit("--store is needed for prepare")
        t0 = time.time()
        n, m = prepare(args.store, work, args.node)
        timings["prepare"] = time.time() - t0
        print(f"corpus.jsonl: {n} documents; entities_{args.node}.jsonl: {m} documents with entities")
    if "graph" in steps:
        if not args.store:
            raise SystemExit("--store is needed for graph")
        cmd = [py, str(HERE / "systems" / "graph_boolean.py"), "--queries", args.queries, "--store", args.store,
               "--work", str(work), "--node", args.node]
        if args.author:
            cmd += ["--author", args.author, "--tag", "graph"]
        timings["graph"] = run(cmd)
    if "bm25" in steps:
        timings["bm25_build"] = run([py, str(HERE / "systems" / "bm25.py"), "build", "--work", str(work), "--text-mode", args.text_mode])
        timings["bm25_search"] = run([py, str(HERE / "systems" / "bm25.py"), "search", "--work", str(work), "--queries", args.queries])
    extra = []
    for k, v in (("--device", args.device), ("--dtype", args.dtype), ("--attn", args.attn)):
        if v:
            extra += [k, v]
    if "embed" in steps:
        timings["embed"] = run([py, str(HERE / "systems" / "pipeline_2026.py"), "embed", "--work", str(work),
                                "--queries", args.queries, "--profile", args.profile, "--text-mode", args.text_mode] + extra)
    if "retrieve" in steps:
        timings["retrieve"] = run([py, str(HERE / "systems" / "pipeline_2026.py"), "retrieve", "--work", str(work),
                                   "--queries", args.queries, "--profile", args.profile])
    if "rerank" in steps:
        timings["rerank"] = run([py, str(HERE / "systems" / "pipeline_2026.py"), "rerank", "--work", str(work),
                                 "--queries", args.queries, "--profile", args.profile, "--text-mode", args.text_mode] + extra)
    if "appendix" in steps and args.appendix:
        cmd = [py, str(HERE / "systems" / "ranked_entities.py"), "--queries", args.queries, "--work", str(work), "--node", args.node]
        if args.store:
            cmd += ["--store", args.store]
        if args.author:
            cmd += ["--author", args.author]
        timings["entities"] = run(cmd)
    if "manifest" in steps:
        runs_dir = work / "runs"
        files = sorted(p for p in runs_dir.glob("*") if p.is_file()) if runs_dir.exists() else []
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip() or None
        except Exception:
            commit = None
        meta = {}
        for name in (f"dense_{profile['name']}/meta.json", f"retrieve_{profile['name']}_meta.json", f"rerank_{profile['name']}_meta.json"):
            p = work / name
            if p.exists():
                meta[name] = read_json(p)
        manifest = {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "profile": profile,
            "queries_file": str(args.queries), "queries_sha256": sha256_file(Path(args.queries)) if Path(args.queries).exists() else None,
            "store": args.store, "node": args.node, "text_mode": args.text_mode, "formula_author": args.author or "consensus/first",
            "corpus_snapshot_date": args.snapshot_date,
            "corpus_sha256": sha256_file(work / "corpus.jsonl") if (work / "corpus.jsonl").exists() else None,
            "n_documents": sum(1 for _ in open(work / "corpus.jsonl", encoding="utf-8")) if (work / "corpus.jsonl").exists() else None,
            "bm25": textprep.bm25_params(), "rrf_k": 60, "channel_depth": 100, "output_depth": 100, "boolean_cap": 1000,
            "filters_applied": "none (date_range in the query file is metadata only)",
            "systems": {"graph": "graph.trec (system 1)", "bm25": "bm25.trec (system 2; also channel 1 of system 3)",
                        "pipeline": "pipeline.trec (system 3)", "dense": "dense.trec (first stage alone)", "fused": "fused.trec (first stages fused, no reranker)",
                        "entities": "entities.trec (ranking over entities, optional)"},
            "step_meta": meta, "timings_seconds": {k: round(v, 1) for k, v in timings.items()},
            "run_files": {p.name: sha256_file(p) for p in files},
            "code_commit": commit, "python": platform.python_version(), "platform": platform.platform(),
        }
        try:
            import torch
            manifest["torch"] = torch.__version__
            manifest["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:
            pass
        try:
            import transformers
            manifest["transformers"] = transformers.__version__
        except Exception:
            pass
        write_json(runs_dir / "manifest.json", manifest)
        print(f"wrote {runs_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
