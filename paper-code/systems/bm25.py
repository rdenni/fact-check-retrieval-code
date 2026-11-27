"""
bm25.py -- system 2: BM25 over the document texts (in the paper's main run the fact-check explanations, --text-mode article), the narrative as the query.

Implementation: bm25s (Lu 2024), Lucene scoring, k1 = 0.9, b = 0.4, English
stop words, Snowball stemmer: the BEIR reference setting (common/textprep.py).
Retrieval is exact over the whole corpus (no approximate index).

The same index and the same function serve as the first channel of the 2026
pipeline, so 'BM25 alone' in the ablation is literally the first channel.

Usage:
  python systems/bm25.py build --work WORK                 # index WORK/corpus.jsonl -> WORK/bm25_index/
  python systems/bm25.py search --work WORK --queries Q.json [--k 100] [--tag bm25]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import kg, textprep  # noqa: E402
from common.queries import load_queries  # noqa: E402
from common.trec import write_run  # noqa: E402


class Bm25Index:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.retriever = None
        self.doc_ids: list[str] = []
        self.stemmer = textprep.get_stemmer()

    @classmethod
    def build(cls, corpus: list[dict], index_dir: Path, text_mode: str = "claim") -> "Bm25Index":
        import bm25s
        self = cls(index_dir)
        n_all = len(corpus)
        corpus = kg.text_corpus(corpus, text_mode)  # explanation mode: documents without an explanation are not indexed
        self.doc_ids = [d["cr_iri"] for d in corpus]
        tokens = textprep.tokenize([kg.doc_text(d, text_mode) for d in corpus], self.stemmer)
        self.retriever = bm25s.BM25(method=textprep.BM25_METHOD, k1=textprep.BM25_K1, b=textprep.BM25_B)
        self.retriever.index(tokens, show_progress=False)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(self.index_dir))
        (self.index_dir / "doc_ids.json").write_text(json.dumps(self.doc_ids), encoding="utf-8")
        (self.index_dir / "params.json").write_text(json.dumps({**textprep.bm25_params(), "text_mode": text_mode, "n_indexed": len(self.doc_ids),
                                                                "n_left_out_no_text": n_all - len(self.doc_ids)}, indent=2), encoding="utf-8")
        return self

    @classmethod
    def load(cls, index_dir: Path) -> "Bm25Index":
        import bm25s
        self = cls(index_dir)
        self.retriever = bm25s.BM25.load(str(self.index_dir))
        self.doc_ids = json.loads((self.index_dir / "doc_ids.json").read_text(encoding="utf-8"))
        return self

    def search(self, query_texts: list[str], k: int) -> list[list[tuple[str, float]]]:
        """For each query text: the top-k (doc_id, score) with score > 0."""
        tokens = textprep.tokenize(query_texts, self.stemmer)
        k = min(k, len(self.doc_ids))
        idx, scores = self.retriever.retrieve(tokens, k=k, show_progress=False)
        out = []
        for row_i, row_s in zip(np.asarray(idx), np.asarray(scores)):
            hits = [(self.doc_ids[int(i)], float(s)) for i, s in zip(row_i, row_s) if s > 0]
            out.append(hits)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--work", required=True)
    b.add_argument("--text-mode", default="article", choices=kg.TEXT_MODES)
    s = sub.add_parser("search")
    s.add_argument("--work", required=True)
    s.add_argument("--queries", required=True)
    s.add_argument("--k", type=int, default=100)
    s.add_argument("--tag", default="bm25")
    args = ap.parse_args()
    work = Path(args.work)
    if args.cmd == "build":
        corpus = kg.load_corpus(work / "corpus.jsonl")
        idx = Bm25Index.build(corpus, work / "bm25_index", args.text_mode)
        print(f"indexed {len(idx.doc_ids)} of {len(corpus)} documents ({args.text_mode}; {len(corpus) - len(idx.doc_ids)} without text left out) "
              f"-> {work / 'bm25_index'}")
    else:
        index = Bm25Index.load(work / "bm25_index")
        queries = load_queries(args.queries)
        hits = index.search([q["free_text"] for q in queries], args.k)
        runs = {q["query_id"]: h for q, h in zip(queries, hits)}
        out = work / "runs" / f"{args.tag}.trec"
        write_run(out, runs, args.tag)
        print(f"wrote {out}: {sum(len(h) for h in hits)} lines for {len(queries)} queries")


if __name__ == "__main__":
    main()
