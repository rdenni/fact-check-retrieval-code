"""
textprep.py -- how text is turned into tokens for BM25.

Rule (model_selection/RULES.md, section 3): the BEIR reference setting.
BEIR (Thakur et al. 2021, Section 3) uses "Anserini with the default Lucene
parameters (k=0.9 and b=0.4)". Anserini's default English analyzer lowercases,
removes Lucene's English stop words and applies the Porter stemmer. We
reproduce that with bm25s (Lu 2024): method "lucene", k1 = 0.9, b = 0.4,
lower-casing, bm25s's English stop-word list, and the Snowball English
stemmer from PyStemmer (Snowball English = "Porter2", the successor of the
Porter stemmer; Anserini's Porter stemmer is the 1980 version. Both are
suffix strippers; the difference is a handful of rules and does not matter
for a baseline).
"""
from __future__ import annotations

BM25_METHOD = "lucene"
BM25_K1 = 0.9
BM25_B = 0.4
STOPWORDS = "english"
STEMMER_LANG = "english"


def get_stemmer():
    import Stemmer
    return Stemmer.Stemmer(STEMMER_LANG)


def tokenize(texts: list[str], stemmer=None, show_progress: bool = False):
    """bm25s tokenizer: lower-case, \\b\\w\\w+\\b tokens, stop words removed,
    stemmed. Returns a bm25s Tokenized object (ids + vocab)."""
    import bm25s
    stemmer = stemmer or get_stemmer()
    return bm25s.tokenize(texts, lower=True, stopwords=STOPWORDS, stemmer=stemmer,
                          return_ids=True, show_progress=show_progress)


def bm25_params() -> dict:
    return {"implementation": "bm25s", "method": BM25_METHOD, "k1": BM25_K1, "b": BM25_B,
            "stopwords": STOPWORDS, "stemmer": f"snowball-{STEMMER_LANG} (PyStemmer)",
            "rule": "BEIR reference setting: Anserini default Lucene parameters k1=0.9, b=0.4 (Thakur et al. 2021, Sec. 3)"}
