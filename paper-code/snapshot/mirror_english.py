"""
mirror_english.py -- download, from the public CimpleKG SPARQL endpoint, every
English fact-check with its claim, its entities and its publisher, into one
N-Triples file. A targeted mirror: only the triples the paper needs, so it
takes about an hour instead of mirroring the whole graph.

Why not the daily dump on GitHub. Checked on 2026-09-25: the release asset
data.claimreview-kg_<date>.nt.gz holds only the fact-checks collected that
day (224 of them) and no schema:mentions at all; the entity links exist only
on the endpoint.

How it works (both queries were tested against the live endpoint on 2026-09-25):
  1. List the IRIs of all ClaimReviews with inLanguage "English", 10,000 per
     request, by SEEK pagination: every request asks for IRIs that sort after
     the last one seen (ORDER BY ?cr, FILTER STR(?cr) > last). Stable, and
     immune to the endpoint's row cap, which breaks OFFSET pagination.
  2. For every batch of --batch IRIs (500), one CONSTRUCT that returns the
     triples of the ClaimReview, of its Claim (itemReviewed) and of its
     publisher (author). 600 IRIs returned 16,367 triples in 16 seconds, so
     there is no 10,000-triple cap on CONSTRUCT.
  3. Every line is cleaned before writing: \\uXXXX escapes inside IRIs are
     turned into the real characters, then the characters an earlier
     store percent-encodes ([]{}()|\\^`"' and blanks) are encoded the same way
     (common/kg.normalise_iri), so that a query address normalised by the
     paper code matches the store.
  Requests are 1 per second (politeness), retried with growing pauses; a
  checkpoint file allows --resume after an interruption.

Outputs in --out: cr_iris.txt, mirror.nt, mirror_state.json, mirror_manifest.json.

Usage:
  python snapshot/mirror_english.py --out SNAPSHOT_DIR [--batch 500] [--sleep 1.0] [--limit 1200] [--resume]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import kg  # noqa: E402

ENDPOINT = "https://data.cimple.eu/sparql"
PAGE = 10000
UCHAR_RE = re.compile(r"\\u([0-9A-Fa-f]{4})|\\U([0-9A-Fa-f]{8})")


def sparql(endpoint: str, query: str, accept: str, timeout: int = 300, retries: int = 6) -> str:
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(endpoint, data={"query": query}, headers={"Accept": accept}, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:  # noqa: BLE001
            last = e
            wait = min(60, 2 ** attempt)
            print(f"    request failed ({type(e).__name__}: {e}); retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"giving up after {retries} attempts: {last}")


def decode_uchar(s: str) -> str:
    return UCHAR_RE.sub(lambda m: chr(int(m.group(1) or m.group(2), 16)), s)


def clean_iri(iri: str) -> str:
    return kg.normalise_iri(decode_uchar(iri))


def clean_line(line: str) -> str | None:
    """One N-Triples line: subject IRI, predicate IRI, object (IRI or literal
    with optional ^^datatype or @lang), final dot. IRIs are cleaned; literals
    are left untouched (a literal may contain '<' or '>' as text, which is why
    a regex over the whole line is not safe)."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    pos = 0
    parts = []
    for _ in range(2):  # subject, predicate
        while pos < len(line) and line[pos] in " \t":
            pos += 1
        if pos >= len(line) or line[pos] != "<":
            return None  # blank-node subject or malformed: not expected from this endpoint
        end = line.index(">", pos)
        parts.append("<" + clean_iri(line[pos + 1:end]) + ">")
        pos = end + 1
    while pos < len(line) and line[pos] in " \t":
        pos += 1
    if pos >= len(line):
        return None
    if line[pos] == "<":
        end = line.index(">", pos)
        parts.append("<" + clean_iri(line[pos + 1:end]) + ">")
    elif line[pos] == '"':
        i = pos + 1
        while i < len(line):
            if line[i] == "\\":
                i += 2
                continue
            if line[i] == '"':
                break
            i += 1
        lit = line[pos:i + 1]
        rest = line[i + 1:].rstrip(" \t.").strip()
        if rest.startswith("^^<") and rest.endswith(">"):
            rest = "^^<" + clean_iri(rest[3:-1]) + ">"
        parts.append(lit + rest)
    else:
        return None
    return " ".join(parts) + " ."


def list_english_crs(endpoint: str, out_path: Path, sleep: float, limit: int | None) -> list[str]:
    if out_path.exists():
        iris = out_path.read_text(encoding="utf-8").split()
        print(f"IRIs already listed: {len(iris)} (from {out_path})")
        return iris[:limit] if limit else iris
    iris: list[str] = []
    last = "http://data.cimple.eu/claim-review/0"
    while True:
        q = (f'SELECT ?cr WHERE {{ ?cr a <{kg.C_CLAIMREVIEW}> ; <{kg.P_LANG}> "English" . '
             f'FILTER(STR(?cr) > "{last}") }} ORDER BY ?cr LIMIT {PAGE}')
        rows = json.loads(sparql(endpoint, q, "application/sparql-results+json"))["results"]["bindings"]
        page = [b["cr"]["value"] for b in rows]
        iris.extend(page)
        print(f"  listed {len(iris):,} English fact-checks", flush=True)
        if len(page) < PAGE or (limit and len(iris) >= limit):
            break
        last = page[-1]
        time.sleep(sleep)
    out_path.write_text("\n".join(iris) + "\n", encoding="utf-8")
    return iris[:limit] if limit else iris


def construct_batch(endpoint: str, iris: list[str]) -> str:
    values = " ".join(f"<{u}>" for u in iris)
    q = (f"CONSTRUCT {{ ?cr ?p ?o . ?claim ?cp ?co . ?org ?op ?oo }} WHERE {{ VALUES ?cr {{ {values} }} ?cr ?p ?o . "
         f"OPTIONAL {{ ?cr <{kg.P_ITEM_REVIEWED}> ?claim . ?claim ?cp ?co }} "
         f"OPTIONAL {{ ?cr <{kg.P_AUTHOR}> ?org . ?org ?op ?oo }} }}")
    return sparql(endpoint, q, "application/n-triples")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default=ENDPOINT)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None, help="only the first N fact-checks (a test)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nt_path, state_path = out / "mirror.nt", out / "mirror_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if (args.resume and state_path.exists()) else {"batches_done": 0, "triples": 0, "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    if not args.resume and nt_path.exists():
        nt_path.unlink()
    iris = list_english_crs(args.endpoint, out / "cr_iris.txt", args.sleep, args.limit)
    batches = [iris[i:i + args.batch] for i in range(0, len(iris), args.batch)]
    print(f"{len(iris):,} fact-checks in {len(batches)} batches of {args.batch}; done so far: {state['batches_done']}", flush=True)
    t0 = time.time()
    with open(nt_path, "a", encoding="utf-8", newline="\n") as fh:
        for i, batch in enumerate(batches):
            if i < state["batches_done"]:
                continue
            raw = construct_batch(args.endpoint, batch)
            n = 0
            for line in raw.splitlines():
                cleaned = clean_line(line)
                if cleaned:
                    fh.write(cleaned + "\n")
                    n += 1
            fh.flush()
            state["batches_done"] = i + 1
            state["triples"] += n
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            rate = (time.time() - t0) / max(1, (i + 1 - (state.get("resumed_from", 0))))
            print(f"  batch {i + 1}/{len(batches)}: {n:,} triples (total {state['triples']:,}); about {rate * (len(batches) - i - 1) / 60:.0f} min left", flush=True)
            time.sleep(args.sleep)
    sha = hashlib.sha256()
    with open(nt_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    manifest = {"endpoint": args.endpoint, "language": "English", "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "started": state["started"], "n_fact_checks_listed": len(iris), "limit": args.limit, "batch": args.batch,
                "triples": state["triples"], "file": nt_path.name, "sha256": sha.hexdigest(),
                "iri_cleaning": "\\uXXXX escapes decoded, then []{}()|\\^`\"' and blanks percent-encoded (common/kg.normalise_iri)",
                "queries": ["SELECT English ClaimReview IRIs, ORDER BY ?cr, seek pagination, 10000 per request",
                            "CONSTRUCT per batch: ?cr ?p ?o ; itemReviewed ?claim ?cp ?co ; author ?org ?op ?oo"]}
    (out / "mirror_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done: {state['triples']:,} triples for {len(iris):,} fact-checks -> {nt_path}; manifest written")


if __name__ == "__main__":
    main()
