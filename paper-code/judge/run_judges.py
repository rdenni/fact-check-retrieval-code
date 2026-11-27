"""
run_judges.py -- the one command that labels the bundle with every judge of a model list. For each judge it either
calls an API that already serves the model (the judge has a base_url, as in judge/judges_api.json, the paper's list)
or starts a local vLLM server on a GPU machine; it labels every pair of the bundle's pool with judge/assess.py
(zero-shot, the prompt of judge/prompts.py), stops the server if it started one, and records what ran; `pack` puts
the results into one zip.

  python judge/run_judges.py run   --config judge/judges_api.json --bundle BUNDLE [--only NAME] [--max-items N]
  python judge/run_judges.py check --out OUT
  python judge/run_judges.py pack  --bundle BUNDLE --out results.zip [--add OUT/env.json ...]

The config is a JSON list of judges: name, model, precision, and optionally base_url and api_key_env (an API: no
server is started, the key is read from that environment variable and never written), revision, serve_args (extra
flags for `vllm serve`), extra (request fields, such as an API's provider routing), max_tokens, concurrency.
--no-serve --base-url URL uses a server that is already running for every judge (the toy test does this with the
fake server). --max-items N labels only N more pairs per judge (the measuring run). A run can be interrupted and
started again: it resumes where it stopped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.jsonl import read_json, read_jsonl, write_json  # noqa: E402
from judge import prompts  # noqa: E402


def same_prompt(man: dict) -> None:
    """The bundle was packed with one prompt (its fingerprint is in MANIFEST.json) and the labels must come from the
    same one. A copy of the code older or newer than the bundle would label with another prompt without anyone
    noticing; this stops the run instead."""
    want, have = man.get("prompt_sha256"), prompts.prompt_fingerprint()
    if want and want != have:
        raise SystemExit(f"the bundle was packed with the prompt sha256 {want[:12]}, "
                         f"but this copy of the code has the prompt sha256 {have[:12]}: "
                         "get the copy of the code that matches the bundle before running")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def quiet(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def hf_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"


def environment() -> dict:
    gpus = quiet(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    versions = {}
    for pkg in ("vllm", "torch", "transformers", "huggingface_hub", "requests"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    cache = hf_cache()
    probe = cache if cache.exists() else Path.home()
    return {"time": now(), "platform": platform.platform(), "python": sys.version.split()[0],
            "gpus": [g.strip() for g in gpus.splitlines() if g.strip()] if gpus else [],
            "versions": versions, "hf_token": bool(os.environ.get("HF_TOKEN") or (cache.parent / "token").exists()),
            "hf_cache": str(cache), "free_disk_gb": round(shutil.disk_usage(probe).free / 1e9, 1)}


def hf_revision(model: str) -> str | None:
    """The commit of the model files actually used: asked to the Hub, else read from the local cache."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model).sha
    except Exception:  # noqa: BLE001
        ref = hf_cache() / f"models--{model.replace('/', '--')}" / "refs" / "main"
        return ref.read_text().strip() if ref.exists() else None


class Server:
    """A vLLM server for one judge, started as a child process, its output in a log file."""

    def __init__(self, judge: dict, gpus: int, port: int, log: Path, ready_timeout: int):
        exe = shutil.which("vllm")
        if not exe:
            raise SystemExit("vllm is not installed here: pip install -r judge/requirements-judge.txt")
        self.cmd = [exe, "serve", judge["model"], "--host", "127.0.0.1", "--port", str(port), "--seed", "0"]
        if judge.get("revision"):
            self.cmd += ["--revision", judge["revision"]]
        if gpus > 1:
            self.cmd += ["--tensor-parallel-size", str(gpus)]
        self.cmd += [str(x) for x in judge.get("serve_args", [])]
        self.base = f"http://127.0.0.1:{port}/v1"
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.fh = open(log, "a", encoding="utf-8")
        print(f"starting: {' '.join(self.cmd)}  (log: {log})", flush=True)
        self.proc = subprocess.Popen(self.cmd, stdout=self.fh, stderr=subprocess.STDOUT)
        import requests
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                self.fh.close()
                tail = "".join(log.read_text(encoding="utf-8", errors="replace").splitlines(True)[-30:])
                raise SystemExit(f"vLLM stopped while starting (exit {self.proc.returncode}); last lines of {log}:\n{tail}")
            try:
                if requests.get(f"{self.base}/models", timeout=5).ok:
                    print(f"server ready after {time.time() - t0:.0f} s", flush=True)
                    return
            except requests.RequestException:
                pass
            if time.time() - t0 > ready_timeout:
                self.stop()
                raise SystemExit(f"vLLM not ready after {ready_timeout} s (download or loading too slow?); see {log}")
            time.sleep(10)

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if not self.fh.closed:
            self.fh.close()


def judges_of(args) -> list[dict]:
    if not Path(args.config).exists():
        raise SystemExit(f"{args.config} does not exist: it lists the judge models for this machine "
                         "(model_selection/RULES.md, section 4; RUN-ON-GPU.md).")
    judges = read_json(args.config)
    if args.only:
        judges = [j for j in judges if j["name"] == args.only]
        if not judges:
            raise SystemExit(f"no judge named {args.only} in {args.config}")
    return judges


def opt(judge: dict, key: str, default):
    return judge.get(key, default)


def assess_args(judge: dict, base: str, concurrency: int, max_items: int | None = None) -> list[str]:
    a = ["--base-url", base, "--model", judge["model"], "--precision", judge["precision"],
         "--concurrency", str(opt(judge, "concurrency", concurrency)), "--max-tokens", str(opt(judge, "max_tokens", 16))]
    if judge.get("extra"):
        a += ["--extra", json.dumps(judge["extra"])]
    if judge.get("api_key_env"):
        a += ["--api-key-env", judge["api_key_env"]]
    if max_items:
        a += ["--max-items", str(max_items)]
    return a


def cmd_check(args):
    env = environment()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "env.json", env)
    print(json.dumps(env, indent=1))
    if not env["gpus"]:
        print("WARNING: no NVIDIA GPU found (nvidia-smi gave nothing)")
    if not env["versions"]["vllm"]:
        print("WARNING: vllm is not installed in this environment")
    if not env["hf_token"]:
        print("WARNING: no Hugging Face token (HF_TOKEN or `huggingface-cli login`); gated models such as Llama cannot download")


def serve_or_not(args, judge: dict, log: Path):
    if args.no_serve:
        if not args.base_url:
            raise SystemExit("--no-serve needs --base-url")
        return None, args.base_url
    if judge.get("base_url"):  # an API that already serves the model: nothing to start
        return None, judge["base_url"]
    srv = Server(judge, args.gpus, args.port, log, args.ready_timeout)
    return srv, srv.base


def cmd_run(args):
    judges = judges_of(args)  # first, so that a missing model list is reported before anything else
    bundle = Path(args.bundle)
    if not (bundle / "MANIFEST.json").exists():
        raise SystemExit(f"{bundle} holds no MANIFEST.json: unzip judge_bundle.zip there first (RUN-ON-GPU.md, step 2)")
    man = read_json(bundle / "MANIFEST.json")
    same_prompt(man)
    (bundle / "judge").mkdir(exist_ok=True)
    common = ["--work", str(bundle), "--queries", str(bundle / man["queries"])]
    for judge in judges:
        name = judge["name"]
        info = {"name": name, "model": judge["model"], "precision": judge["precision"], "config": judge,
                "gpus_flag": args.gpus, "started": now(), "environment": environment()}
        srv, base = serve_or_not(args, judge, bundle / "judge" / "logs" / f"vllm_{name}.log")
        stopped = None
        try:
            info["serve_command"] = srv.cmd if srv else (f"(an API: {base})" if judge.get("base_url") and not args.no_serve
                                                         else f"(server already running at {base})")
            # the exact files are known only when vLLM loads them here; an API does not say which files it serves
            info["revision"] = judge.get("revision") or (hf_revision(judge["model"]) if srv else None)
            r = subprocess.run([sys.executable, str(HERE / "assess.py"), *common,
                                *assess_args(judge, base, args.concurrency, args.max_items), "--name", name])
            if r.returncode != 0:
                stopped = f"judge/assess.py stopped with exit code {r.returncode} (its message is above)"
        finally:
            if srv:
                srv.stop()
        labels = bundle / "judge" / f"labels_llm_{name}.jsonl"
        info["finished"] = now()
        info["labels"] = sum(1 for _ in read_jsonl(labels)) if labels.exists() else 0
        info["pairs_in_pool"] = man["counts"]["pairs"]
        if stopped:
            info["stopped"] = stopped
        write_json(bundle / "judge" / f"run_{name}.json", info)
        missing = info["pairs_in_pool"] - info["labels"]
        print(f"{name}: {info['labels']} labels of {info['pairs_in_pool']} pairs -> {bundle / 'judge'}"
              + (f"; {missing} pairs without a label: run the same command again" if missing and not stopped else ""))
        if stopped:
            raise SystemExit(f"{name}: {stopped}; the labels written so far are kept, and the same command resumes "
                             "once the cause is fixed")


def cmd_pack(args):
    bundle, out = Path(args.bundle), Path(args.out)
    files = [p for p in sorted((bundle / "judge").rglob("*")) if p.is_file()] + [bundle / "MANIFEST.json"]
    files += [Path(p) for p in (args.add or [])]
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            arc = p.relative_to(bundle).as_posix() if p.is_relative_to(bundle) else f"extra/{p.name}"
            z.write(p, arc)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{out}: {len(files)} files, sha256 {h}. Send this file back.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run", "pack"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("--config", required=True)
            p.add_argument("--only", default=None, help="run one judge of the config")
            p.add_argument("--gpus", type=int, default=1, help="GPUs for one server (vLLM tensor parallel size)")
            p.add_argument("--port", type=int, default=8000)
            p.add_argument("--concurrency", type=int, default=64, help="requests in flight, unless the judge's config says otherwise")
            p.add_argument("--ready-timeout", type=int, default=3 * 3600, help="seconds to wait for the download and loading")
            p.add_argument("--no-serve", action="store_true")
            p.add_argument("--base-url", default=None)
            p.add_argument("--max-items", type=int, default=None, help="label only N more pairs per judge (the measuring run)")
            p.add_argument("--bundle", required=True)
        if name == "check":
            p.add_argument("--out", required=True)
        if name == "pack":
            p.add_argument("--bundle", required=True)
            p.add_argument("--out", required=True)
            p.add_argument("--add", nargs="*", help="other files to include, e.g. the env.json written by check")
    args = ap.parse_args()
    {"check": cmd_check, "run": cmd_run, "pack": cmd_pack}[args.cmd](args)


if __name__ == "__main__":
    main()
