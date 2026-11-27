"""
assess.py -- the AI judge: one model labels every pooled (query, document) pair, zero-shot, with the prompt of
judge/prompts.py (model_selection/RULES.md, section 4). No human label is used: the human
labels of the check sample only measure afterwards how good the AI judge is (judge/validate_judge.py).

Reads WORK/pool/pool.jsonl (pool/build_pool.py; in the bundle, the slim copy written by judge/pack_for_judge.py)
and the query file. Appends to WORK/judge/labels_llm_<name>.jsonl, one line per pair:
  query_id, doc_id, label ('y' / 'n'), model, precision, base_url, prompt_sha256, claim_truncated,
  seconds, prompt_tokens, completion_tokens, reasoning_tokens, cost, finish_reason, provider, served_model,
  generation_id, raw (the reply, cut to 200 characters), ts.
(provider, served_model, generation_id and cost come from an API that routes to several providers, OpenRouter;
they are null for a local vLLM server.)
Re-running resumes: pairs already labelled are skipped. A reply that holds neither YES nor NO is not written and is
asked again on the next run (the console counts them). An answer that no retry can change (a wrong key, no credit
left, a provider choice the server does not have: judge/llm_client.py FATAL_STATUS) stops the whole run at once.
--concurrency N keeps N requests in flight (the server batches them); 1 = one at a time.
--max-tokens and --extra serve models that reason before answering and carry an API's routing fields
(judge/judges_api.json, model_selection/RULES.md, section 4).
--api-key-env NAME reads the API key from that environment variable (the key is never written anywhere).

Usage:
  python judge/assess.py --work WORK --queries Q.json --base-url https://openrouter.ai/api/v1 \
      --api-key-env OPENROUTER_API_KEY --model openai/gpt-oss-120b --precision mxfp4 --name gptoss120b \
      --concurrency 16 --max-tokens 4096 --extra '{"reasoning": {"effort": "medium"}}'
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonl import read_jsonl, append_jsonl  # noqa: E402
from common.queries import load_queries  # noqa: E402
from judge.llm_client import ChatClient, FatalAPIError, parse_yes_no  # noqa: E402
from judge import prompts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="the work folder, or the unzipped bundle (judge/pack_for_judge.py)")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True, help="as served: bf16, fp8, mxfp4 ... (goes into every label)")
    ap.add_argument("--name", required=True, help="short name for the output file, e.g. gptoss120b")
    ap.add_argument("--max-items", type=int, default=None, help="label only the first N pairs still to do (tests)")
    ap.add_argument("--concurrency", type=int, default=1, help="requests in flight at once (vLLM batches them); 1 = one at a time")
    ap.add_argument("--max-tokens", type=int, default=16, help="room for the reply; a model that reasons first needs a few thousand")
    ap.add_argument("--extra", default=None, help='extra request fields as JSON, e.g. \'{"reasoning": {"effort": "medium"}}\'')
    ap.add_argument("--api-key-env", default=None, help="the environment variable that holds the API key (none for a local server)")
    args = ap.parse_args()
    extra = json.loads(args.extra) if args.extra else None
    api_key = "none"
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            raise SystemExit(f"the environment variable {args.api_key_env} is empty or not set: set it to the API key "
                             "(judge/RUN-WITH-API.md) and open a new terminal")
    work = Path(args.work)
    queries = {q["query_id"]: q for q in load_queries(args.queries)}
    pool = list(read_jsonl(work / "pool" / "pool.jsonl"))
    missing_q = sorted({r["query_id"] for r in pool} - set(queries))
    if missing_q:
        raise SystemExit(f"the pool has queries that the query file lacks: {missing_q[:5]}")
    out_path = work / "judge" / f"labels_llm_{args.name}.jsonl"
    done = {(r["query_id"], r["doc_id"]) for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in pool if (r["query_id"], r["doc_id"]) not in done]
    if args.max_items:
        todo = todo[:args.max_items]
    fingerprint = prompts.prompt_fingerprint()
    print(f"pairs in the pool {len(pool)}, already labelled {len(done)}, to label now {len(todo)} -> {out_path} "
          f"(prompt sha256 {fingerprint[:12]}, concurrency {args.concurrency})")
    client = ChatClient(args.base_url, args.model, api_key=api_key)
    stop = threading.Event()  # set by the first fatal answer: the requests not yet sent are skipped
    fatal: list[str] = []

    def judge_one(r: dict):
        if stop.is_set():
            return r, False, None, "skipped"
        prompt, cut = prompts.judge_prompt(queries[r["query_id"]], r)
        try:
            return r, cut, client.chat_full(None, prompt, max_tokens=args.max_tokens, extra=extra), None
        except FatalAPIError as e:
            stop.set()
            fatal.append(str(e))
            return r, cut, None, "fatal"
        except Exception as e:  # noqa: BLE001
            return r, cut, None, f"server error: {e}"

    from tqdm import tqdm
    counts, failed = Counter(), Counter()
    p_tok, c_tok, costs = [], [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = [ex.submit(judge_one, r) for r in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="pair"):
            r, cut, out, err = fut.result()
            if err in ("fatal", "skipped"):
                failed["not sent: the run stopped" if err == "skipped" else "refused by the server"] += 1
                continue
            if err:
                failed["server error"] += 1
                tqdm.write(f"  {err[:160]} ({r['query_id']} {r['doc_id'][-20:]})")
                continue
            lab = parse_yes_no(out["text"])
            if lab is None:
                why = "reply cut by --max-tokens" if out["finish_reason"] == "length" else "reply without YES or NO"
                failed[why] += 1
                tqdm.write(f"  {why}: {out['text'][:80]!r}")
                continue
            usage = out["usage"]
            reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            append_jsonl(out_path, {"query_id": r["query_id"], "doc_id": r["doc_id"], "label": lab,
                                    "model": args.model, "precision": args.precision, "base_url": args.base_url,
                                    "prompt_sha256": fingerprint, "claim_truncated": cut,
                                    "seconds": round(out["seconds"], 2), "prompt_tokens": usage.get("prompt_tokens"),
                                    "completion_tokens": usage.get("completion_tokens"), "reasoning_tokens": reasoning,
                                    "cost": usage.get("cost"), "finish_reason": out["finish_reason"],
                                    "provider": out.get("provider"), "served_model": out.get("served_model"),
                                    "generation_id": out.get("generation_id"),
                                    "raw": out["text"].strip()[:200], "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
            counts[lab] += 1
            if usage.get("prompt_tokens") is not None:
                p_tok.append(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                c_tok.append(usage["completion_tokens"])
            if isinstance(usage.get("cost"), (int, float)):
                costs.append(usage["cost"])
    secs = time.time() - t0
    n_ok = counts["y"] + counts["n"]
    print(f"labelled YES={counts['y']} NO={counts['n']}; not labelled {sum(failed.values())} {dict(failed) if failed else ''}"
          + ("; run the same command again to ask those again" if failed and not fatal else ""))
    if n_ok:
        print(f"{secs:.0f} s, {n_ok / secs:.2f} pairs per second" +
              (f"; mean prompt tokens {statistics.mean(p_tok):.0f}" if p_tok else "") +
              (f", mean reply tokens {statistics.mean(c_tok):.1f}" if c_tok else ""))
    if costs:
        mean = statistics.mean(costs)
        print(f"cost reported by the server: {sum(costs):.4f} dollars for {len(costs)} labels, {mean:.6f} a label; "
              f"the {len(pool) - len(done) - n_ok} pairs still to label would cost about {mean * (len(pool) - len(done) - n_ok):.2f} dollars")
    if fatal:
        raise SystemExit(f"STOPPED: the server refused a request in a way no retry can change: {fatal[0]}")


if __name__ == "__main__":
    main()
