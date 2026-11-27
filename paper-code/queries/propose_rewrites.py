"""
propose_rewrites.py -- the model's proposal of the query text for every drawn
query: the same local model, settings and logging as the
drafting (draft_needs.py), the prompt queries/prompts/rewrite_needs.txt; the
reviewer then reads the passage and accepts or corrects every proposal in the
rewrite sheet (draw_queries.py rewrite-sheet --proposals), and import-rewrite
checks the final text and counts the changed proposals.

The query is the claim as one self-contained sentence, the way fact-checkers
write the claims they check ("clearer, context-independent, and
self-contained", Shaar et al. 2020, section 3); a place or a date enters the
sentence only when it cannot be understood without them (which attack, which
election), because the relevance guideline makes neither decide relevance
(pool/GUIDELINE.md). Decided before any row was
rewritten; until then the program appended the place and the date in brackets.
Why a proposal and not a free rewrite: a model drafts the sentence and its
place and date (shown to the reviewer as a hint) and a person checks every
row, as in the TREC 2025 RAG track (section 2: the
sub-narratives drafted by GPT-4.1 were "systematically reviewed" and
"post-edited" by people) and TREC DL 2023 (model-written queries selected by
NIST assessors).

Input: --queries (queries_drawn.json or queries_pilot.json), the period and
the verdict words of every query from WORK/drafting/reviewed.jsonl.
Output: WORK/drafting/rewrite_proposals_<queries stem>.jsonl, one line per
query: query_id, cluster_id, proposal (the model's statement; compose()),
hint (the place and the date of the model's parts, each kept only if the
passage names it; where_when()), doubt (the model's, plus every part the
program left out), parts, period, parsed, the raw reply, seconds,
thinking_chars, and what decides the reply (model, precision,
prompt_sha256, reasoning, max_tokens, request_sha256, Ollama version and model
digest); --resume reuses a reply made from exactly the same request.

Usage:
  python queries/propose_rewrites.py --queries WORK/queries_drawn.json --work WORK --base-url http://localhost:11434/v1 \
         --model qwen3-14b-16k --precision q4_K_M --no-think [--resume]
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.jsonl import read_jsonl, write_jsonl, read_json  # noqa: E402
from judge.llm_client import ChatClient, extract_json  # noqa: E402
from draft_needs import load_prompt, request_key, server_info  # noqa: E402
PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "rewrite_needs.txt"
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def compose(obj: dict, passage: str, period: str) -> tuple[str, list[str], str]:
    """The query from the model's parts: the statement alone, as one self-contained sentence (
    before any row was rewritten: relevance does not depend on the place or the date, so they are not appended in
    brackets; they enter the sentence only when it cannot be understood without them, and the reviewer decides that).
    The place and the date are returned as a hint for the reviewer (where_when), with the notes of what was left out."""
    s = str(obj.get("statement", "") or "").strip().rstrip(".").strip()
    parts, notes = where_when(obj, passage, period, s)
    return (s + "." if s else ""), notes, ", ".join(parts)


def where_when(obj: dict, passage: str, period: str, s: str) -> tuple[list[str], list[str]]:
    """The place and the date of the model's parts, each kept only if the passage contains it (a place or date the
    passage does not name is left out and noted), the date only for a claim about something that happened at a time,
    its year taken from the review's period when the passage gives none (no date when the period is empty too),
    nothing repeated that the statement already says. In an earlier version these were appended to the statement in
    brackets; now they are only shown to the reviewer as a hint."""
    low_p, low_s = passage.lower(), s.lower()
    parts, notes = [], []
    place = str(obj.get("place", "") or "").strip()
    if place:
        if place.lower() not in low_p:
            notes.append(f"place '{place}' is not in the passage: left out")
        elif re.sub(r"^the ", "", place.lower()) not in low_s:
            parts.append(place)
    if obj.get("time_bound") in (True, "true", "True"):
        when = str(obj.get("when", "") or "").strip()
        if when and when.lower() not in low_p:
            notes.append(f"date '{when}' is not in the passage: left out")
            when = ""
        if when and when.lower() in low_s:  # the statement already says it ("on the 9/11 anniversary"): only a missing year
            when = ""
        if not YEAR.search(when):
            when = f"{when} {period}".strip() if YEAR.search(period or "") else ""
        if when and when.lower() not in low_s:
            parts.append(when)
    return parts, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True, help="bf16 | q4_K_M | ... (written into every proposal and the paper)")
    ap.add_argument("--no-think", action="store_true", help="thinking off, as in the drafting (reasoning_effort none and /no_think)")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--resume", action="store_true", help="reuse the replies of identical requests already in the output file")
    ap.add_argument("--prompt", type=Path, default=PROMPT_FILE)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    system_tpl, user_tpl, prompt_sha = load_prompt(args.prompt)
    out = args.out or args.work / "drafting" / f"rewrite_proposals_{args.queries.stem}.jsonl"
    reviewed = {r["cluster_id"]: r for r in read_jsonl(args.work / "drafting" / "reviewed.jsonl")}
    qs = read_json(args.queries)
    client = ChatClient(args.base_url, args.model, timeout=args.timeout)
    server = server_info(args.base_url, args.model)
    reasoning = "none" if args.no_think else "default"
    extra = {"reasoning_effort": "none"} if args.no_think else None
    old = {r["request_sha256"]: r for r in read_jsonl(out)} if args.resume and out.exists() else {}
    for r in old.values():  # one software state per file, as in the drafting
        diff = {k: (r.get(k), v) for k, v in server.items() if r.get(k) not in (None, v)}
        if diff:
            print(f"[propose] STOP: {r['query_id']} was answered with {diff} (saved, now); start a new file instead of resuming")
            return 2
    print(f"[propose] {len(qs)} queries; model {args.model} ({args.precision}); prompt sha256 {prompt_sha[:12]}; server {server or 'not Ollama'}")
    rows = []
    for q in qs:
        p = q["provenance"]
        r = reviewed.get(p["cluster_id"], {})
        sys_msg = system_tpl
        user = user_tpl.format(claim=p["claim"], passage=p["passage"], verdict_words=r.get("verdict_words", ""))
        if args.no_think:
            user += "\n/no_think"
        req = request_key(args.model, reasoning, args.max_tokens, sys_msg, user)
        if req in old:  # the same request answered before: its reply is reused and joined again by the current compose()
            o = old[req]
            prop, notes, hint = compose(extract_json(o["reply"]) or {}, p["passage"], r.get("period", ""))
            rows.append({**o, "proposal": prop, "hint": hint, "parsed": bool(prop),
                         "doubt": "; ".join([str((extract_json(o["reply"]) or {}).get("doubt", "") or "").strip()] + notes).strip("; ")})
            continue
        reply, secs = client.chat(sys_msg, user, max_tokens=args.max_tokens, json_mode=True, extra=extra)
        thinking = len(client.last_message.get("reasoning") or "") + sum(len(t) for t in re.findall(r"<think>(.*?)</think>", reply, flags=re.S))
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
        obj = extract_json(reply) or {}
        proposal, notes, hint = compose(obj, p["passage"], r.get("period", ""))
        rows.append({"query_id": q["query_id"], "cluster_id": p["cluster_id"], "proposal": proposal, "hint": hint,
                     "doubt": "; ".join([str(obj.get("doubt", "") or "").strip()] + notes).strip("; "),
                     "parts": {k: obj.get(k) for k in ("statement", "place", "when", "time_bound")}, "period": r.get("period", ""),
                     "parsed": bool(proposal), "reply": reply, "seconds": round(secs, 1), "thinking_chars": thinking,
                     "time": dt.datetime.now().isoformat(timespec="seconds"), "model": args.model, "precision": args.precision,
                     "prompt_sha256": prompt_sha, "reasoning": reasoning, "max_tokens": args.max_tokens, "request_sha256": req, **server})
        write_jsonl(out, rows)  # after every reply, so that an interrupted run can be resumed
        print(f"[propose] {q['query_id']:22s} {secs:5.1f}s  {rows[-1]['proposal'][:90]}")
    write_jsonl(out, rows)
    bad = [r["query_id"] for r in rows if not r["parsed"]]
    doubts = sum(1 for r in rows if r["doubt"])
    thought = sum(1 for r in rows if r.get("thinking_chars"))
    print(f"[propose] {len(rows)} proposals in {out}; unparsed {len(bad)}{' (' + ', '.join(bad) + ')' if bad else ''}; "
          f"with a doubt {doubts}; replies with thinking {thought}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
