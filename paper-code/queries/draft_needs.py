"""
draft_needs.py -- the drafting step: a language model reads the tier-1
documents of one (country, year) cell, chunk by chunk, and lists every false
or misleading statement the text says circulated, written as the statement
itself (no attribution, no verdict), with the passage quoted verbatim and the
passage's own words that call it false (verdict_words). The output is the
candidate list from which the queries are drawn at random after review.

The model is any server that speaks the OpenAI chat API (llama.cpp's
llama-server, Ollama or vLLM): the same client
as the judge (judge/llm_client.py), temperature 0 and a fixed seed. The prompt
is the file queries/prompts/draft_needs.txt, released with the paper; its
sha256 goes into every candidate.

Input:  the cell files of prepare_cells.py (reports/cells/<folder>-<year>.md)
Output: WORK/drafting/candidates_<cell>.jsonl, one line per candidate with its
        provenance (cell, document label, tier, url, file sha256, chunk index,
        model, precision, prompt sha256) and the two checks the script makes:
        passage_found (the quoted passage is in the chunk, verbatim up to
        whitespace), verdict_found (the verdict words are in the chunk) and
        country_fit (the model's own field); the "need" sentence is built by
        the script from the statement ("fact-checks about the claim that ...");
        WORK/drafting/raw/<cell>/<doc>-<chunk>.json, the raw replies, so that a
        run can be resumed and audited; WORK/drafting/draft_log.jsonl.

Every raw reply carries request_sha256, the hash of everything that decides
the reply (model, thinking setting, budget, the two messages with the chunk);
--resume reuses a saved reply only for the same hash, and a request already
answered for another cell of the same country (a document read by several
cells; the model is told the document's own years) is copied, not re-asked.
A reply cut off at the budget is retried at once, once, with a frequency
penalty. The Ollama version and the model's digest go into every raw reply.

Usage:
  python queries/draft_needs.py --cells reports/cells --work WORK --base-url http://localhost:8080/v1
         --model Qwen3-8B-Q4_K_M --precision q4_K_M [--only us-2020,india-2023] [--chunk-chars 20000]
         [--overlap-chars 1500] [--max-tokens 8000] [--timeout 900] [--limit-chunks N] [--resume]
  With Ollama: --base-url http://localhost:11434/v1 --model qwen3-14b-16k (the model made from
  queries/ollama/Modelfile-qwen3-14b-16k, which sets the 16,384-token context)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import unicodedata
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.jsonl import write_jsonl, append_jsonl  # noqa: E402
from judge.llm_client import ChatClient, extract_json  # noqa: E402

HERE = Path(__file__).resolve().parent
PROMPT_FILE = HERE / "prompts" / "draft_needs.txt"
TOPICS = {"elections", "politics", "health", "science-climate", "security-conflict", "economy", "migration", "society-religion", "other"}


def load_prompt(path: Path = PROMPT_FILE) -> tuple[str, str, str]:
    """The prompt file holds the system part and the user template; returns (system, user_template, sha256)."""
    text = path.read_text(encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    system = text.split("=== SYSTEM ===", 1)[1].split("=== USER ===", 1)[0].strip()
    user = text.split("=== USER ===", 1)[1].strip()
    return system, user, sha


def parse_cell(path: Path) -> tuple[dict, list[dict]]:
    """A cell file of prepare_cells.py: the header line and one dict per document (label, meta, text)."""
    text = path.read_text(encoding="utf-8")
    head = re.match(r"# Cell (.+?) (\d{4}): (\d+) queries, (\d+) documents", text)
    if not head:
        raise ValueError(f"not a cell file: {path}")
    cell = {"name": path.stem, "country": head.group(1), "year": int(head.group(2)), "queries": int(head.group(3))}
    docs = []
    for block in text.split("\n\n---\n\n## DOCUMENT ")[1:]:
        lines = block.split("\n")
        label = lines[0].strip()
        meta: dict = {}
        body_start = 1
        for i, l in enumerate(lines[1:], start=1):
            if l.startswith(("tier:", "file:", "note:")):
                for part in l.split(" | "):
                    k, _, v = part.partition(":")
                    meta[k.strip().replace(" ", "_")] = v.strip()
                body_start = i + 1
            elif l.strip() == "":
                body_start = i + 1
                break
        docs.append({"label": label, "meta": meta, "text": "\n".join(lines[body_start:]).strip()})
    return cell, docs


def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    """Pieces of about chunk_chars, cut at paragraph ends, each starting with the tail of the previous one."""
    if len(text) <= chunk_chars:
        return [text]
    paras = [p for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 1 > chunk_chars:
            chunks.append(cur)
            cur = cur[-overlap_chars:] + "\n" + p if overlap_chars else p
        else:
            cur = (cur + "\n" + p) if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def candidates_of(reply: str) -> tuple[list, bool]:
    """The candidate list of a reply. (items, complete): complete is False when the JSON could not be parsed as a
    whole (an answer cut off at the token budget, or empty) and the items are the complete objects salvaged from
    the "candidates" array, each one still a full JSON object; an empty reply gives ([], False)."""
    obj = extract_json(reply) or {}
    items = obj.get("candidates") if isinstance(obj, dict) else None
    if isinstance(items, list):
        return items, True
    text = reply or ""
    start = text.find("[", text.find("candidates")) if "candidates" in text else -1
    if start < 0:
        return [], False
    out, depth, i, obj_start = [], 0, start + 1, None
    in_str, esc = False, False
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    out.append(json.loads(text[obj_start:i + 1]))
                except json.JSONDecodeError:
                    pass
                obj_start = None
        elif ch == "]" and depth == 0:
            break
        i += 1
    return out, False


def norm_text(s: str) -> str:
    """Letters and digits only, lower case, one space between words: the comparison that decides passage_found and
    verdict_found. It ignores what a model cannot copy reliably: curly versus straight quotes, en and em dashes,
    the replacement character of a bad PDF glyph, accents (NFKC) and spacing. The words must still be the text's."""
    s = unicodedata.normalize("NFKC", s).replace("\ufffd", " ")  # a broken glyph counts as a non-letter, like the letter it hid
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def dehyphenate(s: str) -> str:
    """Join a word that the PDF split at the end of a line ("misinfor-\nmation" -> "misinformation"): the model quotes
    the word whole, the text file has it in two pieces. Only a hyphen at the end of a line is removed."""
    return re.sub(r"(\w)-[ \t]*\r?\n[ \t]*(\w)", r"\1\2", s)


def found_in(quote: str, chunk_norm: str) -> tuple[bool, bool]:
    """(found, elided): the quote is in the chunk, verbatim under norm_text; a quote with an ellipsis is accepted when
    every part of at least 20 characters is found in order (the parts are verbatim, the gap is marked)."""
    parts = [x for x in re.split(r"\.\.\.|\u2026", quote) if norm_text(x)]
    if len(parts) <= 1:
        return norm_text(quote) in chunk_norm, False
    pos = 0
    for part in parts:
        n = norm_text(part)
        if len(n) < 20:
            return False, True
        i = chunk_norm.find(n, pos)
        if i < 0:
            return False, True
        pos = i + len(n)
    return True, True


_RO_CACHE: dict[str, str] = {}


def reading_order_norm(reports: Path | None, file_rel: str) -> str:
    """The normalised reading-order text (pdftotext without -layout) of the source PDF, computed once per file; "" for
    an HTML source or when pdftotext is missing. Used as a second place where a quote may be found: a quote is
    verbatim if it is in the chunk the model read OR anywhere in the reading-order text of the same document (on the
    single-column reports, page headers and figure numbers that -layout puts inside sentences are elsewhere in
    reading order: the Long Fuse 132 -> 148 of 163 quotes, First Draft 87 -> 100 of 100, none lost)."""
    if not reports or not file_rel or not file_rel.lower().endswith(".pdf"):
        return ""
    if file_rel not in _RO_CACHE:
        import subprocess
        import tempfile
        from fetch_pages import find_pdftotext
        exe = find_pdftotext()
        text = ""
        if exe and (reports / file_rel).exists():
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "ro.txt"
                subprocess.run([exe, "-q", str(reports / file_rel), str(out)], check=False)
                if out.exists():
                    text = out.read_text(encoding="utf-8", errors="replace")
        _RO_CACHE[file_rel] = norm_text(dehyphenate(text))
    return _RO_CACHE[file_rel]


def validate(cand: dict, chunk: str, doc_norm: str = "") -> dict | None:
    """Keep the fields the prompt asks for, in the types it asks for; build the need sentence from the statement;
    mark whether the passage and the verdict words are really in the chunk."""
    if not isinstance(cand, dict):
        return None
    statement = str(cand.get("statement") or cand.get("claim") or "").strip().rstrip(".")
    passage = str(cand.get("passage", "")).strip()
    if not statement or not passage:
        return None
    verdict = str(cand.get("verdict_words", "")).strip()
    ents = cand.get("entities") or []
    if isinstance(ents, str):
        ents = [e.strip() for e in ents.split(",") if e.strip()]
    topic = str(cand.get("topic", "other")).strip().lower()
    chunk_norm = norm_text(chunk)
    p_found, p_elided = found_in(passage, chunk_norm)
    if not p_found:  # the same check on the text with line-end hyphens joined
        chunk_norm_h = norm_text(dehyphenate(chunk))
        p_found, p_elided = found_in(passage, chunk_norm_h)
    else:
        chunk_norm_h = chunk_norm
    p_where = "chunk" if p_found else ""
    if not p_found and doc_norm:  # anywhere in the reading-order text of the same source document
        p_found, p_elided = found_in(passage, doc_norm)
        p_where = "document (reading order)" if p_found else ""
    return {"need": "fact-checks about the claim that " + statement[:400], "claim": statement[:400],
            "spreader": str(cand.get("spreader", "")).strip()[:200], "verdict_words": verdict[:300], "passage": passage[:600],
            "entities": [str(e).strip() for e in ents if str(e).strip()],
            "period": str(cand.get("period", "")).strip(), "topic": topic if topic in TOPICS else "other",
            "country_fit": str(cand.get("country_fit", "unclear")).strip().lower(),
            "passage_found": p_found, "passage_elided": p_elided, "passage_where": p_where,
            "verdict_found": bool(verdict) and (found_in(verdict, chunk_norm)[0] or found_in(verdict, chunk_norm_h)[0]
                                                or (bool(doc_norm) and found_in(verdict, doc_norm)[0]))}


def server_info(base_url: str, model: str) -> dict:
    """The server's version and the model's digest, written into every raw reply so that the paper can name the exact
    software and weights. Ollama only (its /api/version and /api/tags); {} for another server or when unreachable."""
    import requests
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        version = requests.get(f"{root}/api/version", timeout=10).json().get("version", "")
        tags = requests.get(f"{root}/api/tags", timeout=10).json().get("models", [])
        digest = next((m.get("digest", "") for m in tags if m.get("name") in (model, f"{model}:latest")), "")
        return {"ollama_version": version, "model_digest": digest}
    except Exception:  # noqa: BLE001
        return {}


def request_key(model: str, reasoning: str, max_tokens: int, system: str, user: str) -> str:
    """The sha256 of everything that decides the reply: model, thinking setting, answer budget and the two messages
    (which hold the prompt, the country, the document's years, its label and the chunk's text)."""
    return hashlib.sha256(json.dumps([model, reasoning, max_tokens, system, user], ensure_ascii=False).encode("utf-8")).hexdigest()


def is_final(raw: dict) -> bool:
    """A saved reply is final when its JSON is whole, or when it is already the one penalised retry."""
    return candidates_of(raw["reply"])[1] or bool(raw.get("frequency_penalty"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True, help="bf16 | q4_K_M | q8_0 ... (written into every candidate and the paper)")
    ap.add_argument("--only", default="", help="comma-separated cell names (file stems)")
    ap.add_argument("--chunk-chars", type=int, default=20000)
    ap.add_argument("--overlap-chars", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=8000, help="answer budget; a long report chunk can hold 40 statements, about 5,000 tokens of JSON")
    ap.add_argument("--retry-penalty", type=float, default=0.5, help="frequency penalty of the one retry of a cut-off reply")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds to wait for one reply before retrying (a long chunk on the laptop can need 10 minutes)")
    ap.add_argument("--limit-chunks", type=int, default=0, help="stop after N chunks per cell (tests)")
    ap.add_argument("--resume", action="store_true", help="skip chunks whose raw reply exists and was made from exactly the same request (request_sha256)")
    ap.add_argument("--prompt", type=Path, default=PROMPT_FILE)
    ap.add_argument("--no-think", action="store_true", help="turn the model's thinking off: reasoning_effort none in the request (Ollama) and /no_think in the message (Qwen3's soft switch; the 14B ignores it alone)")
    ap.add_argument("--dry-run", action="store_true", help="count documents, chunks and characters per cell; call nothing")
    args = ap.parse_args()

    system, user_tpl, prompt_sha = load_prompt(args.prompt)
    if args.dry_run:
        only = {s.strip() for s in args.only.split(",") if s.strip()}
        tot_calls = tot_chars = 0
        for cell_path in sorted(p for p in args.cells.glob("*.md") if p.stem != "coverage" and (not only or p.stem in only)):
            cell, docs = parse_cell(cell_path)
            n_chunks = sum(len(chunk_text(d["text"], args.chunk_chars, args.overlap_chars)) for d in docs)
            chars = sum(len(d["text"]) for d in docs)
            tot_calls += n_chunks
            tot_chars += chars
            print(f"[draft] {cell['name']:22s} {len(docs):3d} docs {n_chunks:4d} calls {chars:9,d} chars")
        print(f"[draft] total {tot_calls} calls, {tot_chars:,} characters (about {tot_chars // 4:,} input tokens)")
        return 0
    # the quote check re-extracts PDFs in reading order: it must use the program that made the saved texts
    from fetch_pages import find_pdftotext, pdftotext_version, VERSION_FILE
    needs_pdf = any("file: " in l and ".pdf |" in l for p in args.cells.glob("*.md") for l in p.read_text(encoding="utf-8").splitlines()
                    if l.startswith("file: "))
    pdf_version = ""
    if needs_pdf:
        exe = find_pdftotext()
        vf = args.cells.parent / VERSION_FILE
        made_by = vf.read_text(encoding="utf-8").strip() if vf.exists() else ""
        pdf_version = pdftotext_version(exe) if exe else ""
        if not exe or pdf_version != made_by:
            print(f"[draft] STOP: the texts were made by '{made_by or 'unknown'}' and this terminal's pdftotext is "
                  f"'{pdf_version or 'missing'}' ({exe}); run fetch_pages.py --retext and prepare_cells.py in this terminal, "
                  f"or set PDFTOTEXT to the program that made them")
            return 2
    client = ChatClient(args.base_url, args.model, timeout=args.timeout)
    server = {**server_info(args.base_url, args.model), **({"pdftotext": pdf_version} if pdf_version else {})}
    reasoning = "none" if args.no_think else "default"
    extra = {"reasoning_effort": "none"} if args.no_think else None
    out_dir = args.work / "drafting"
    raw_dir = out_dir / "raw"
    # one software state per run: a reply saved by another Ollama version, model digest or pdftotext stops the run
    # (Ollama on Windows updates itself when its tray icon is told to; a resumed run after that would mix two versions)
    for f in sorted(raw_dir.rglob("*.json")) if args.resume and raw_dir.exists() else []:
        old_raw = json.loads(f.read_text(encoding="utf-8"))
        diff = {k: (old_raw.get(k), v) for k, v in server.items() if old_raw.get(k) not in (None, v)}
        if diff:
            print(f"[draft] STOP: {f.relative_to(raw_dir)} was made with {diff} (saved, now); the run would mix two software "
                  f"versions. Put the drafting folder aside and run from zero, or go back to the earlier version.")
            return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    cells = sorted(p for p in args.cells.glob("*.md") if p.stem != "coverage" and (not only or p.stem in only))
    print(f"[draft] {len(cells)} cells; model {args.model} ({args.precision}); prompt sha256 {prompt_sha[:12]}; server {server or 'not Ollama'}")
    answered: dict[str, Path] = {}  # request_sha256 -> the raw file that holds its final reply, in this run

    for cell_path in cells:
        cell, docs = parse_cell(cell_path)
        cands, n_calls, n_cut, n_think, t_cell = [], 0, 0, 0, time.time()
        (raw_dir / cell["name"]).mkdir(parents=True, exist_ok=True)
        for d in docs:
            chunks = chunk_text(d["text"], args.chunk_chars, args.overlap_chars)
            for ci, chunk in enumerate(chunks):
                if args.limit_chunks and n_calls >= args.limit_chunks:
                    break
                raw_path = raw_dir / cell["name"] / f"{d['label']}-{ci:03d}.json"
                raw = json.loads(raw_path.read_text(encoding="utf-8")) if args.resume and raw_path.exists() else None
                chunk_sha = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                # the model is told the years the DOCUMENT covers, not the cell's year: a report on 2020-2025 is read
                # by six cells, and the same request then gets the same answer, asked once (below)
                period = d["meta"].get("years") or str(cell["year"])
                sys_msg = system.replace("{country}", cell["country"])  # str.format cannot be used on it because of the JSON braces
                user = user_tpl.format(country=cell["country"], period=period, source=d["label"], text=chunk)
                if args.no_think:
                    user += "\n/no_think"
                req_sha = request_key(args.model, reasoning, args.max_tokens, sys_msg, user)
                record = {"cell": cell["name"], "doc": d["label"], "chunk": ci, "chars": len(chunk), "period": period,
                          "model": args.model, "precision": args.precision, "prompt_sha256": prompt_sha, "reasoning": reasoning,
                          "max_tokens": args.max_tokens, "chunk_sha256": chunk_sha, "request_sha256": req_sha, **server}
                if raw and raw.get("request_sha256") == req_sha and is_final(raw):
                    reply = raw["reply"]  # this exact request was answered before: whole, or already retried once
                elif req_sha in answered:
                    # the same document chunk was answered for another cell of this country: copied, not asked again
                    prev = json.loads(answered[req_sha].read_text(encoding="utf-8"))
                    reply = prev["reply"]
                    raw_path.write_text(json.dumps({**record, "frequency_penalty": prev.get("frequency_penalty"),
                                                    "first_reply_cut": prev.get("first_reply_cut"), "seconds": 0.0,
                                                    "thinking_chars": prev.get("thinking_chars"), "completion_tokens": prev.get("completion_tokens"),
                                                    "copied_from": answered[req_sha].relative_to(raw_dir).as_posix(), "reply": reply},
                                                   indent=1), encoding="utf-8")
                else:
                    # a cut-off reply (a loop, or a list longer than the budget) is retried at once with a frequency
                    # penalty, which lowers the probability of tokens the reply has already used; the retry is final,
                    # salvaged object by object if it is still cut; a saved cut reply of this request goes straight to it
                    penalty = args.retry_penalty if raw and raw.get("request_sha256") == req_sha else None
                    first_cut = {"chars": len(raw["reply"]), "seconds": raw.get("seconds")} if penalty else None
                    while True:
                        call_extra = {**(extra or {}), **({"frequency_penalty": penalty} if penalty else {})} or None
                        reply, secs = client.chat(sys_msg, user, max_tokens=args.max_tokens, json_mode=True, extra=call_extra)
                        # thinking is switched off by the chat template (an empty <think></think> is written before the
                        # answer); this measures it for every reply rather than trusting it: Ollama's reasoning field
                        # plus any <think> block left in the content
                        thinking = len(client.last_message.get("reasoning") or "") + sum(
                            len(t) for t in re.findall(r"<think>(.*?)</think>", reply, flags=re.S))
                        tokens = client.last_usage.get("completion_tokens")
                        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()  # Qwen3 thinking block, if any
                        if thinking and args.no_think:
                            n_think += 1
                            append_jsonl(out_dir / "draft_log.jsonl", {"time": dt.datetime.now().isoformat(timespec="seconds"), "cell": cell["name"],
                                                                        "doc": d["label"], "chunk": ci, "event": "thinking appeared", "chars": thinking})
                        append_jsonl(out_dir / "draft_log.jsonl", {"time": dt.datetime.now().isoformat(timespec="seconds"), "cell": cell["name"],
                                                                    "doc": d["label"], "chunk": ci, "chars": len(chunk), "seconds": round(secs, 1),
                                                                    "frequency_penalty": penalty})
                        if penalty or candidates_of(reply)[1]:
                            break
                        first_cut = {"chars": len(reply), "seconds": round(secs, 1)}
                        penalty = args.retry_penalty
                    raw_path.write_text(json.dumps({**record, "frequency_penalty": penalty, "first_reply_cut": first_cut,
                                                    "seconds": round(secs, 1), "thinking_chars": thinking, "completion_tokens": tokens,
                                                    "reply": reply}, indent=1), encoding="utf-8")
                answered.setdefault(req_sha, raw_path)
                n_calls += 1
                items, complete = candidates_of(reply)
                if not complete:
                    n_cut += 1
                    append_jsonl(out_dir / "draft_log.jsonl", {"time": dt.datetime.now().isoformat(timespec="seconds"), "cell": cell["name"],
                                                                "doc": d["label"], "chunk": ci, "event": "reply cut off or empty", "salvaged": len(items)})
                doc_norm = reading_order_norm(args.cells.parent, d["meta"].get("file", ""))
                for k, item in enumerate(items):
                    v = validate(item, chunk, doc_norm)
                    if v is None:
                        continue
                    cands.append({"candidate_id": f"{cell['name']}:{d['label']}:{ci:03d}:{k:02d}", "cell": cell["name"],
                                  "country": cell["country"], "year": cell["year"], "doc": d["label"],
                                  "tier": d["meta"].get("tier", ""), "publisher_type": d["meta"].get("publisher_type", ""),
                                  "url": d["meta"].get("url", ""), "file": d["meta"].get("file", ""), "sha256": d["meta"].get("sha256", ""),
                                  "chunk": ci, **v, "reply_complete": complete, "model": args.model, "precision": args.precision,
                                  "prompt_sha256": prompt_sha})
        n = write_jsonl(out_dir / f"candidates_{cell['name']}.jsonl", cands)
        found = sum(1 for c in cands if c["passage_found"])
        print(f"[draft] {cell['name']:22s} {len(docs):3d} docs {n_calls:4d} chunks {n:5d} candidates ({found} with the passage found"
              f"{'; ' + str(n_cut) + ' replies cut off' if n_cut else ''}"
              f"{'; WARNING ' + str(n_think) + ' replies with thinking' if n_think else ''}) "
              f"in {time.time() - t_cell:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
