"""
entity_catalog.py -- the list of Wikipedia/DBpedia entity pages that occur in the
archive, with how many fact-checks mention each one; plus a search over that list.

For the person who runs the import and for the failure analysis only. It is NOT given
to the writers of the Boolean formulas: they formulate from the narrative and
English Wikipedia alone, as a user of the system would, and a formula tuned to
the archive's vocabulary would test an easier system than the one the KG papers
describe. The person who runs the import uses it to
read the import reports (which terms match nothing and why) and the failure
protocol uses the same entity links to count ENTITY_MISMATCH.

Usage:
  python queries/entity_catalog.py build  --work WORK [--node cr] [--out WORK/stats/entity_catalog.csv]
  python queries/entity_catalog.py search --catalog WORK/stats/entity_catalog.csv ukrain [biolab ...] [--top 25]
  python queries/entity_catalog.py exact  --catalog WORK/stats/entity_catalog.csv "Biological warfare" ["Ukraine" ...]

The CSV columns: wikipedia_title, dbpedia_uri, n_docs, pct_docs, first_year,
last_year, n_publishers. 'search' finds titles that contain every given word
(case does not matter) and prints them by n_docs; 'exact' checks whole titles,
which is what the import of the writers' sheets does (import_formulas.py).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonl import read_jsonl  # noqa: E402
from common.kg import normalise_iri  # noqa: E402

DBPEDIA = "http://dbpedia.org/resource/"
COLUMNS = ["wikipedia_title", "dbpedia_uri", "n_docs", "pct_docs", "first_year", "last_year", "n_publishers"]


def title_from_uri(uri: str) -> str:
    """'http://dbpedia.org/resource/People%27s_Movement' -> "People's Movement".
    Everything after '/resource/' is the title, slashes included: the store writes 'HIV/AIDS' and
    'Radio_Free_Europe/Radio_Liberty' with a bare slash (an earlier version kept only the part
    after the last slash, so 'HIV/AIDS' read as 'AIDS' and the redirect name 'Mass media/India' as 'India')."""
    tail = uri.split("/resource/", 1)[1] if "/resource/" in uri else uri.rsplit("/", 1)[-1]
    return unquote(tail).replace("_", " ")


def uri_from_title(title: str) -> str:
    """The inverse, as the store encodes it: spaces to underscores, first letter
    upper case (Wikipedia titles always start with a capital), percent-encoding
    of the characters the store encodes."""
    t = title.strip().replace(" ", "_")
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return normalise_iri(DBPEDIA + t)


def norm_key(uri: str) -> str:
    """One spelling for matching: decoded, underscores as spaces, lower case."""
    return title_from_uri(uri).casefold()


def build(work: Path, node: str, out: Path) -> int:
    corpus = {d["cr_iri"]: d for d in read_jsonl(work / "corpus.jsonl")}
    ents_path = work / f"entities_{node}.jsonl"
    if not ents_path.exists():
        raise SystemExit(f"{ents_path} not found: run run_all.py --steps prepare --node {node} first")
    n_docs: dict[str, int] = defaultdict(int)
    years: dict[str, list] = defaultdict(lambda: [None, None])
    pubs: dict[str, set] = defaultdict(set)
    n_universe = len(corpus)
    for r in read_jsonl(ents_path):
        d = corpus.get(r["cr_iri"])
        if d is None:  # not in the document universe (removed by a cleaning rule)
            continue
        y = (d.get("date_published") or "")[:4]
        y = y if y.isdigit() else None
        for e in set(r.get("entities") or []):
            n_docs[e] += 1
            if y:
                lo, hi = years[e]
                years[e] = [y if lo is None or y < lo else lo, y if hi is None or y > hi else hi]
            if d.get("author_name"):
                pubs[e].add(d["author_name"])
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(n_docs.items(), key=lambda kv: (-kv[1], kv[0]))
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:  # utf-8-sig so Excel opens it with the right letters
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for e, n in rows:
            lo, hi = years[e]
            w.writerow([title_from_uri(e), e, n, f"{100 * n / max(n_universe, 1):.3f}", lo or "", hi or "", len(pubs[e])])
    return len(rows)


def load_catalog(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["n_docs"] = int(r["n_docs"])
        r["_key"] = norm_key(r["dbpedia_uri"])
    return rows


def search(rows: list[dict], words: list[str], top: int) -> list[dict]:
    ws = [w.casefold() for w in words if w.strip()]
    hits = [r for r in rows if all(w in r["_key"] for w in ws)]
    hits.sort(key=lambda r: (-r["n_docs"], r["wikipedia_title"]))
    return hits[:top]


def exact(rows: list[dict], titles: list[str]) -> dict[str, dict | None]:
    by_key = {r["_key"]: r for r in rows}
    return {t: by_key.get(norm_key(uri_from_title(t))) for t in titles}


def print_rows(hits: list[dict]) -> None:
    if not hits:
        print("  (no entity page matches)")
        return
    print(f"  {'n_docs':>7}  {'years':<9}  {'publ.':>5}  title  |  address")
    for r in hits:
        yrs = f"{r['first_year']}-{r['last_year']}" if r["first_year"] else "undated"
        print(f"  {r['n_docs']:>7}  {yrs:<9}  {r['n_publishers']:>5}  {r['wikipedia_title']}  |  {r['dbpedia_uri']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--work", required=True)
    b.add_argument("--node", default="cr", help="which node's entity links: cr (explanation, the paper's choice) or claim")
    b.add_argument("--out", default=None)
    s = sub.add_parser("search")
    s.add_argument("--catalog", required=True)
    s.add_argument("words", nargs="+")
    s.add_argument("--top", type=int, default=25)
    x = sub.add_parser("exact")
    x.add_argument("--catalog", required=True)
    x.add_argument("titles", nargs="+")
    args = ap.parse_args()
    if args.cmd == "build":
        work = Path(args.work)
        out = Path(args.out) if args.out else work / "stats" / "entity_catalog.csv"
        n = build(work, args.node, out)
        print(f"entity catalog: {n} distinct entity pages on the {args.node} node -> {out}")
        return
    rows = load_catalog(Path(args.catalog))
    if args.cmd == "search":
        hits = search(rows, args.words, args.top)
        print(f"titles containing {args.words} (top {args.top} of {len(search(rows, args.words, 10**9))}):")
        print_rows(hits)
    else:
        for t, r in exact(rows, args.titles).items():
            if r is None:
                print(f"  NOT IN THE ARCHIVE: '{t}' -> {uri_from_title(t)} (a formula using it matches nothing)")
            else:
                print(f"  {r['n_docs']:>7} docs  '{t}' -> {r['dbpedia_uri']}")


if __name__ == "__main__":
    main()
