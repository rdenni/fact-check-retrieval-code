"""
queries.py -- the query file of the paper (version 6) and its checks.

One query (JSON object) has:
  query_id        stable id (string)
  free_text       the narrative in plain words; the query of BM25 and of the pipeline
  background      optional: who or what a name of the query is, written from the
                  source report; shown to the judges only (judges' sheet, AI judge), never
                  searched by any system and not shown to the writers
  query_type      narrative | episode | single_claim
  stratum         {"country": ..., "year": ...}   where the query comes from (metadata)
  source          {"report": ..., "url_or_doi": ..., "date": ...}  (metadata)
  filters         kept from version 5 as metadata; NEVER applied by any system
  entity_groups   kept from version 5 as metadata
  boolean_formulas: list of
      {"author_id": "annotator_a" | "annotator_b" (the toy test: writer_a, writer_b) | "consensus",
       "formula": "e1 AND (e2 OR e3)",
       "entities": {"e1": {"label": ..., "wikipedia_title": ..., "wikidata_qid": ..., "dbpedia_uri": ...}, ...},
       "not_expressible": ["text", ...],   concepts the author could not turn into entities
       "minutes": number or null,
       "refinements": number or null}

Which formula the graph system runs: --author on the command line; default
'main' if present (the assigned writer's formula, copied by
queries/import_formulas.py assign-main), else 'consensus', else the first
formula in the list.
"""
from __future__ import annotations

from pathlib import Path

from .boolean import parse, terms, FormulaError
from .jsonl import read_json

AUTHOR_DEFAULT_ORDER = ("main", "consensus")


def load_queries(path: str | Path) -> list[dict]:
    qs = read_json(path)
    if not isinstance(qs, list):
        raise ValueError("query file must be a JSON list")
    seen = set()
    for q in qs:
        if "query_id" not in q or "free_text" not in q:
            raise ValueError(f"query without query_id or free_text: {q}")
        if q["query_id"] in seen:
            raise ValueError(f"duplicate query_id {q['query_id']}")
        seen.add(q["query_id"])
    return qs


def pick_formula(query: dict, author_id: str | None = None) -> dict | None:
    fs = query.get("boolean_formulas") or []
    if not fs:
        return None
    if author_id:
        for f in fs:
            if f.get("author_id") == author_id:
                return f
        return None
    for a in AUTHOR_DEFAULT_ORDER:
        for f in fs:
            if f.get("author_id") == a:
                return f
    return fs[0]


def check_formula(f: dict) -> list[str]:
    """Return a list of problems (empty = fine)."""
    problems = []
    try:
        node = parse(f.get("formula", ""))
    except FormulaError as e:
        return [f"cannot parse: {e}"]
    ents = f.get("entities") or {}
    for t in terms(node):
        if t not in ents:
            problems.append(f"term {t} has no entry in entities")
        elif not (ents[t].get("dbpedia_uri") or "").startswith("http"):
            problems.append(f"term {t} ({ents[t].get('label')}) has no dbpedia_uri: it can never match")
    return problems


def check_queries(qs: list[dict]) -> dict[str, list[str]]:
    """query_id -> problems, only for queries with problems."""
    out = {}
    for q in qs:
        probs = []
        for f in q.get("boolean_formulas") or []:
            for p in check_formula(f):
                probs.append(f"[{f.get('author_id')}] {p}")
        if probs:
            out[q["query_id"]] = probs
    return out


def term_uri_map(f: dict) -> dict[str, str]:
    """entity key -> dbpedia_uri (percent-encoded like the store's IRIs, see
    kg.normalise_iri), only for keys that have one."""
    from .kg import normalise_iri
    return {k: normalise_iri(v["dbpedia_uri"]) for k, v in (f.get("entities") or {}).items() if v.get("dbpedia_uri")}


def wikipedia_title_from_dbpedia(uri: str) -> str:
    from urllib.parse import unquote
    if not uri:
        return ""
    tail = uri.split("/resource/", 1)[1] if "/resource/" in uri else uri.rsplit("/", 1)[-1]  # 'HIV/AIDS' keeps its slash
    return unquote(tail).replace("_", " ")
