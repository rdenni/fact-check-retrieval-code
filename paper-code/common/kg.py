"""
kg.py -- everything the paper code reads from the CimpleKG Oxigraph store.

The store is the cleaned English snapshot built by snapshot/build_store.py. This module never writes to it.

Vocabulary used below (plain words):
  ClaimReview (CR)  = one fact-checking explanation. Its IRI is the document id.
  Claim             = the sentence the explanation checks (schema:itemReviewed).
  claim_text        = schema:text of the Claim.
  article_text      = schema:text of the ClaimReview: the explanation. The text the paper's text systems read.
  mentions          = schema:mentions, the DBpedia entities the KG builders
                      linked in a text. They exist on the Claim node (from the
                      claim sentence) and on the CR node (from the explanation).

The corpus snapshot (corpus.jsonl) fixes the document universe once, so every
system, the pool, and the metrics agree on which documents exist.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator

from .jsonl import read_jsonl, write_jsonl

SCHEMA = "http://schema.org/"
CIMPLE = "http://data.cimple.eu/ontology#"

P_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
C_CLAIMREVIEW = SCHEMA + "ClaimReview"
P_ITEM_REVIEWED = SCHEMA + "itemReviewed"
P_TEXT = SCHEMA + "text"
P_MENTIONS = SCHEMA + "mentions"
P_DATE = SCHEMA + "datePublished"
P_AUTHOR = SCHEMA + "author"
P_NAME = SCHEMA + "name"
P_LANG = SCHEMA + "inLanguage"
P_URL = SCHEMA + "url"
P_RATING = CIMPLE + "normalizedReviewRating"

NODE_CHOICES = ("claim", "cr", "union")

# Which text the text systems read. "claim": the claim sentence only.
# "article": the fact-check explanation's text alone (schema:text on the ClaimReview node), which is exactly the text
# CimpleKG's explanation-level entities come from, so the graph and the text systems search the same text and differ only
# in how they search; THE PAPER'S MAIN RUNS. A document without
# an explanation has no text in this mode and is left out of the text index (text_corpus below), as it has no entity for
# the graph either. "claim+article": the claim sentence followed by the explanation, what a search engine over the whole
# fact-check reads; these runs are pooled and judged by the AI judges, and the paper does not report them.
TEXT_MODES = ("claim", "claim+article", "article")


def doc_text(d: dict, mode: str) -> str:
    if mode == "claim":
        return d["claim_text"]
    art = (d.get("article_text") or "").strip()
    if mode == "article":
        return art
    if mode == "claim+article":
        return d["claim_text"] + ("\n" + art if art else "")
    raise ValueError(f"text mode must be one of {TEXT_MODES}, got {mode!r}")


def text_corpus(corpus: list[dict], mode: str) -> list[dict]:
    """The documents a text system indexes in this mode: those with some text. In 'article' mode the documents
    without an explanation are left out, so no text system can return them (the graph cannot either); in the other modes
    every document has its claim, so nothing is left out."""
    return [d for d in corpus if doc_text(d, mode).strip()]

# The snapshot percent-encodes these characters inside every IRI before loading
# the store (Oxigraph rejects them). A DBpedia
# URI written by a human, e.g. .../Mercury_(planet), must get the same
# treatment before it is matched against the store, or it never matches.
_IRI_BAD_CHARS = '[](){}|\\^`"\' \t'
_IRI_ENCODE = {c: f"%{ord(c):02X}" for c in _IRI_BAD_CHARS}


def normalise_iri(uri: str) -> str:
    """Percent-encode the characters the store's IRIs have encoded. Idempotent."""
    return "".join(_IRI_ENCODE.get(c, c) for c in (uri or "").strip())


def open_store(path: str | Path):
    """The store, read only: nothing here writes to it, and a read-only store takes no lock, so several scripts can read
    it at the same time (a store opened for writing locks it for every other process)."""
    from pyoxigraph import Store
    return Store.read_only(str(path))


# ---------------------------------------------------------------------------
# Corpus snapshot
# ---------------------------------------------------------------------------
def iter_claim_reviews(store) -> Iterator[dict]:
    """Yield one record per ClaimReview that has a Claim with a non-empty text.

    If a CR has several Claims or a Claim has several texts, the record keeps
    the smallest (claim_iri, text) pair in sorting order and sets
    n_claim_texts to the number of pairs, so the choice is deterministic and
    visible.
    """
    q = f"""
    SELECT ?cr ?claim ?text ?article ?date ?author_name ?rating ?language WHERE {{
      ?cr <{P_TYPE}> <{C_CLAIMREVIEW}> ;
          <{P_ITEM_REVIEWED}> ?claim .
      ?claim <{P_TEXT}> ?text .
      OPTIONAL {{ ?cr <{P_TEXT}> ?article }}
      OPTIONAL {{ ?cr <{P_DATE}> ?date }}
      OPTIONAL {{ ?cr <{P_AUTHOR}> ?au . ?au <{P_NAME}> ?author_name }}
      OPTIONAL {{ ?cr <{P_RATING}> ?rating }}
      OPTIONAL {{ ?cr <{P_LANG}> ?language }}
    }}
    """
    by_cr: dict[str, list[dict]] = defaultdict(list)
    for row in store.query(q):
        text = row["text"].value if row["text"] else ""
        if not text or not text.strip():
            continue
        by_cr[row["cr"].value].append({
            "claim_iri": row["claim"].value,
            "claim_text": text.strip(),
            "article_text": row["article"].value.strip() if row["article"] else None,
            "date_published": row["date"].value if row["date"] else None,
            "author_name": row["author_name"].value if row["author_name"] else None,
            "rating": row["rating"].value if row["rating"] else None,
            "language": row["language"].value if row["language"] else None,
        })
    for cr_iri in sorted(by_cr):
        recs = sorted(by_cr[cr_iri], key=lambda r: (r["claim_iri"], r["claim_text"], r["article_text"] or "", r["date_published"] or "",
                                                    r["author_name"] or "", r["rating"] or "", r["language"] or ""))
        r = dict(recs[0])
        r["cr_iri"] = cr_iri
        r["n_claim_texts"] = len(recs)
        yield r


def build_corpus_snapshot(store, out_path: str | Path) -> int:
    """Write corpus.jsonl: one line per document with cr_iri, claim_iri,
    claim_text, article_text (the explanation body, None when absent), date_published (as in the KG, usually yyyy-mm-dd...),
    author_name, rating, language, n_claim_texts. Returns the count."""
    return write_jsonl(out_path, iter_claim_reviews(store))


def load_corpus(path: str | Path) -> list[dict]:
    return list(read_jsonl(path))


def date_key(date_str: str | None) -> str:
    """First 10 characters (yyyy-mm-dd) or '' when missing, for sorting."""
    return (date_str or "")[:10]


# ---------------------------------------------------------------------------
# Entity mentions
# ---------------------------------------------------------------------------
def _entity_query(node: str, entity_values: str | None = None) -> str:
    """SPARQL that returns (?cr, ?e) pairs for the requested node.

    node = "claim": entities of the Claim node (the claim sentence). This is
           what the ClaimsKG and CimpleKG papers query in their examples.
    node = "cr":    entities of the ClaimReview node (the explanation). This is
           what the paper queries: on the English part of CimpleKG every
           Claim node is linked to the same entity, "Mars", so the Claim
           node cannot be used.
    node = "union": either.
    entity_values: optional 'VALUES ?e { <..> <..> }' block to restrict.
    """
    values = f"VALUES ?e {{ {entity_values} }}" if entity_values else ""
    claim_pat = f"?cr <{P_ITEM_REVIEWED}> ?claim . ?claim <{P_MENTIONS}> ?e ."
    cr_pat = f"?cr <{P_MENTIONS}> ?e ."
    if node == "claim":
        body = claim_pat
    elif node == "cr":
        body = cr_pat
    elif node == "union":
        body = f"{{ {claim_pat} }} UNION {{ {cr_pat} }}"
    else:
        raise ValueError(f"node must be one of {NODE_CHOICES}, got {node!r}")
    return f"""
    SELECT DISTINCT ?cr ?e WHERE {{
      {values}
      ?cr <{P_TYPE}> <{C_CLAIMREVIEW}> .
      {body}
    }}
    """


def crs_mentioning(store, entity_uri: str, node: str = "claim") -> set[str]:
    """One SPARQL query for one entity term: the set of CR IRIs whose chosen
    node mentions the entity. This is the query the graph system runs per
    formula term (paper: 'evaluated with SPARQL')."""
    q = _entity_query(node, f"<{entity_uri}>")
    return {row["cr"].value for row in store.query(q)}


def build_entity_index(store, node: str, out_path: str | Path) -> int:
    """Dump every (cr_iri, entity_uri) pair of the chosen node to a JSONL file
    {cr_iri, entities: [...]}. Used by the failure protocol, the statistics,
    the 'index' backend of the graph system, and the tests. Returns the number
    of CRs with at least one entity."""
    ents: dict[str, set[str]] = defaultdict(set)
    for row in store.query(_entity_query(node)):
        ents[row["cr"].value].add(row["e"].value)
    rows = ({"cr_iri": cr, "entities": sorted(es)} for cr, es in sorted(ents.items()))
    return write_jsonl(out_path, rows)


def load_entity_index(path: str | Path) -> dict[str, set[str]]:
    return {r["cr_iri"]: set(r["entities"]) for r in read_jsonl(path)}


def invert_entity_index(index: dict[str, set[str]]) -> dict[str, set[str]]:
    inv: dict[str, set[str]] = defaultdict(set)
    for cr, es in index.items():
        for e in es:
            inv[e].add(cr)
    return inv


def count_crs(store) -> int:
    q = f"SELECT (COUNT(DISTINCT ?cr) AS ?n) WHERE {{ ?cr <{P_TYPE}> <{C_CLAIMREVIEW}> }}"
    rows = list(store.query(q))
    return int(rows[0]["n"].value) if rows else 0
