"""
llm_client.py -- one small client for any server that speaks the OpenAI chat
API, such as vLLM (base URL http://localhost:8000/v1), Ollama (base URL http://localhost:11434/v1) or OpenRouter
(base URL https://openrouter.ai/api/v1, with an API key):
Deterministic settings: temperature 0, a fixed seed, a small max_tokens.
"""
from __future__ import annotations

import json
import re
import time

import requests

# HTTP answers that asking again cannot change: a wrong key (401), no credit left (402), a refused account (403), a
# model or a provider choice the server does not have (404).
FATAL_STATUS = (401, 402, 403, 404)


class FatalAPIError(RuntimeError):
    """The server refused the request for a reason that holds for every request (FATAL_STATUS): stop, do not retry."""


def provider_of(body: dict) -> str | None:
    """The provider that answered, from an OpenRouter reply: a top-level "provider" field if present, else the endpoint
    marked "selected" in openrouter_metadata.endpoints.available (OpenRouter's API specification). None for a local
    server, or when the reply carries neither (the generation id then still identifies the request)."""
    if body.get("provider"):
        return body["provider"]
    meta = body.get("openrouter_metadata") or {}
    for e in (meta.get("endpoints") or {}).get("available") or []:
        if e.get("selected"):
            return e.get("provider")
    return None


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: str = "none", timeout: int = 180,
                 seed: int = 42, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.model, self.api_key, self.timeout, self.seed, self.retries = model, api_key, timeout, seed, retries
        self.last_message: dict = {}  # the whole message of the last reply (Ollama puts any thinking in "reasoning")
        self.last_usage: dict = {}    # its token counts (prompt_tokens, completion_tokens)

    def chat_full(self, system: str, user: str, max_tokens: int = 64, json_mode: bool = False,
                  extra: dict | None = None) -> dict:
        """One call; returns text, seconds, the whole message, the token counts, why the reply stopped and, from an
        API that routes to several providers (OpenRouter), who answered: provider, served_model, generation_id.
        Keeps no state on the client, so several threads can share one client (judge/assess.py --concurrency).
        Raises FatalAPIError at once on an answer in FATAL_STATUS; other failures are retried."""
        # system may be None: the AI judge's prompt is one user message (judge/prompts.py)
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        payload = {"model": self.model, "temperature": 0.0, "seed": self.seed, "max_tokens": max_tokens, "messages": messages}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if extra:
            # e.g. {"reasoning_effort": "none"} (Ollama, Qwen3 thinking off), or OpenRouter's {"reasoning": {...},
            # "provider": {...}} (judge/judges_api.json)
            payload.update(extra)
        last = None
        t0 = time.time()
        for attempt in range(self.retries):
            pause = 1.5 * (attempt + 1)
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout,
                                  headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
                if r.status_code in FATAL_STATUS:
                    raise FatalAPIError(f"HTTP {r.status_code} from {self.base_url}: {r.text[:300]}")
                if r.status_code == 429:  # too many requests: wait longer before asking again
                    pause = 10.0 * (attempt + 1)
                r.raise_for_status()
                body = r.json()
                if "choices" not in body:  # an error object instead of a reply
                    raise RuntimeError(f"no reply in the answer: {str(body.get('error', body))[:300]}")
                choice = body["choices"][0]
                message = choice["message"]
                return {"text": message.get("content") or "", "seconds": time.time() - t0, "message": message,
                        "usage": body.get("usage") or {}, "finish_reason": choice.get("finish_reason"),
                        "provider": provider_of(body), "served_model": body.get("model"), "generation_id": body.get("id")}
            except FatalAPIError:
                raise
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(pause)
        raise RuntimeError(f"chat call failed after {self.retries} attempts: {last}")

    def chat(self, system: str, user: str, max_tokens: int = 64, json_mode: bool = False,
             extra: dict | None = None) -> tuple[str, float]:
        out = self.chat_full(system, user, max_tokens=max_tokens, json_mode=json_mode, extra=extra)
        self.last_message, self.last_usage = out["message"], out["usage"]
        return out["text"], out["seconds"]


def extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        obj = json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_label(text: str) -> str | None:
    obj = extract_json(text)
    if not obj or not isinstance(obj.get("label"), str):
        return None
    lab = obj["label"].strip().lower()
    if lab in ("relevant", "yes", "y", "true", "1"):
        return "y"
    if lab in ("not relevant", "not_relevant", "no", "n", "false", "0", "irrelevant"):
        return "n"
    if "not" in lab and "relevant" in lab:
        return "n"
    if lab.startswith("rel"):
        return "y"
    return None


def parse_yes_no(text: str) -> str | None:
    """The AI judge's answer (judge/prompts.py asks for one word, YES or NO): 'y', 'n', or None when the reply holds
    neither word. The first YES or NO in the reply counts ('YES.', 'Answer: NO' are read); a JSON reply with a
    "label" field is accepted too."""
    lab = parse_label(text)
    if lab:
        return lab
    for word in re.findall(r"[A-Za-z]+", (text or "").upper()):
        if word == "YES":
            return "y"
        if word == "NO":
            return "n"
    return None
