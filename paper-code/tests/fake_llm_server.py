"""fake_llm_server.py -- a tiny OpenAI-compatible server for testing judge/*.py without a model.
It answers the AI judge's prompt (judge/prompts.py) with YES when the claim shares two long words with the
narrative, else NO, and the drafting and rewrite prompts of queries/ with fixed JSON.
Usage: python tests/fake_llm_server.py --port 8123"""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


TOY_KEY = "toy-key"  # the key the toy test puts in the environment for the API judge


class H(BaseHTTPRequestHandler):
    def refuse(self, status: int, message: str):
        out = json.dumps({"error": {"code": status, "message": message}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        model = str(body.get("model"))
        # an API like OpenRouter: model "fake-api" needs the toy key; model "fake-nocredit" is refused as an account
        # without credit (HTTP 402), which must stop the judge at once
        if model == "fake-nocredit":
            return self.refuse(402, "Insufficient credits")
        if model == "fake-api" and self.headers.get("Authorization") != f"Bearer {TOY_KEY}":
            return self.refuse(401, "No auth credentials found")
        user = body["messages"][-1]["content"]
        if "PASSAGE:" in user and "Now list the false or misleading statements" in user:
            # the drafting prompt (queries/draft_needs.py): answer with one candidate per sentence that contains
            # the word "claim", quoting that sentence verbatim as the passage and the word "claim" as the verdict
            # words, so both checks of the script pass
            passage = user.split("PASSAGE:", 1)[1].split("\nNow list the false or misleading statements", 1)[0]
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage) if "claim" in s.lower() and 40 <= len(s.strip()) <= 300]
            cands = [{"statement": s.split("claim")[-1].strip(" .:,")[:120], "spreader": "social media users", "verdict_words": "claim",
                      "passage": s, "entities": [w for w in re.findall(r"\b[A-Z][a-z]+", s)][:5], "period": "", "topic": "politics",
                      "country_fit": "yes"} for s in sents]
            content = json.dumps({"candidates": cands})
        elif "Now prepare the parts of the query" in user:
            # the rewrite proposal (queries/propose_rewrites.py): the claim itself as the statement, no place, a time-bound
            # claim with no date of its own (so the program adds the period's year)
            claim = re.search(r"CLAIM: (.*)", user).group(1).strip()
            content = json.dumps({"statement": claim, "place": "", "when": "", "time_bound": True, "doubt": ""})
        else:
            # the AI judge's prompt (judge/prompts.py): YES when the claim shares two long words with the narrative
            # the case comes after the guideline's examples, which also have "Query:" lines: take the last one
            narrative = re.findall(r"^Query: (.*)$", user, re.M)[-1].lower().split()
            claim = re.findall(r"^Claim: (.*)$", user, re.M)[-1].lower()
            hit = sum(1 for w in narrative if len(w) > 4 and w[:5] in claim) >= 2
            content = "YES" if hit else "NO"
        usage = {"prompt_tokens": len(" ".join(m["content"] for m in body["messages"]).split()), "completion_tokens": len(content.split())}
        reply = {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                 "model": body.get("model"), "usage": usage}
        if model == "fake-api":  # what OpenRouter adds: who answered (in its documented place), the generation id, the cost
            usage["cost"] = 0.0001
            reply.update({"id": "gen-toy", "openrouter_metadata": {
                "endpoints": {"available": [{"provider": "OtherProvider", "selected": False},
                                            {"provider": "FakeProvider", "selected": True}], "total": 2},
                "summary": "available=2, selected=FakeProvider"}})
        out = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()  # answers several requests at once (judge --concurrency)
