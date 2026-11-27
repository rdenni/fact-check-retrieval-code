"""
check_source_overlap.py -- the mechanical check behind condition (i) of the
source rule: "a source is not made of the archive's
documents". For every saved report source it counts the links that point to
fact-checks of the archive (exact URL match against the store's ClaimReview
URLs) and the links that point to the domains of the archive's publishers,
and it says whether the source's own site is an archive publisher.

A thematic review that cites a few archive fact-checks as examples is fine
(the narratives are the review's, the links are evidence); a page whose links
are mostly archive fact-checks is an item list and belongs to tier X. The
numbers make that judgement checkable instead of felt.

Usage:
  python queries/check_source_overlap.py --mirror C:/snapshot-2026-10/mirror.nt
         --corpus C:/snapshot-2026-10/final/work/corpus.jsonl
         [--list queries/report_sources.csv] [--reports reports] [--out reports/source-overlap.csv]

The mirror holds one <schema:url> triple per ClaimReview (all languages); the
corpus says which ClaimReviews are in the English store. Links are read from
the saved .html (href attributes) and from the .txt of PDFs (bare URLs).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
URL_TRIPLE = re.compile(r"^<([^>]+)> <http://schema\.org/url> <([^>]+)> \.$")
HREF = re.compile(r'href="(https?://[^"#]+)', re.I)
BARE = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
TRACKING = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid"}


def norm(url: str) -> str:
    """One spelling per address: lower-case host without www, no tracking parameters, no fragment, no trailing slash."""
    try:
        p = urllib.parse.urlsplit(html.unescape(url.strip()))
    except ValueError:
        return url
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    query = "&".join(q for q in p.query.split("&") if q and q.split("=")[0] not in TRACKING)
    path = p.path.rstrip("/")
    return host + path + ("?" + query if query else "")


def domain(url: str) -> str:
    try:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
    except ValueError:  # a malformed address in a page (an unbalanced bracket): no domain
        return ""
    return host[4:] if host.startswith("www.") else host


def archive_urls(mirror: Path, corpus: Path) -> tuple[set[str], Counter]:
    keep = set()
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            keep.add(json.loads(line)["cr_iri"])
    urls: set[str] = set()
    domains: Counter = Counter()
    with open(mirror, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "<http://schema.org/url>" not in line:
                continue
            m = URL_TRIPLE.match(line.strip())
            if not m or m.group(1) not in keep:
                continue
            urls.add(norm(m.group(2)))
            domains[domain(m.group(2))] += 1
    return urls, domains


def links_of(path_html: Path | None, path_txt: Path | None) -> set[str]:
    links: set[str] = set()
    if path_html and path_html.exists():
        links.update(HREF.findall(path_html.read_text(encoding="utf-8", errors="replace")))
    if path_txt and path_txt.exists():
        links.update(BARE.findall(path_txt.read_text(encoding="utf-8", errors="replace")))
    return {html.unescape(u).rstrip(".,;") for u in links}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mirror", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--list", type=Path, default=HERE / "report_sources.csv")
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-domain-count", type=int, default=20,
                    help="a domain counts as an archive publisher when at least this many archive URLs use it")
    args = ap.parse_args()
    out = args.out or (args.reports / "source-overlap.csv")

    urls, dom_counts = archive_urls(args.mirror, args.corpus)
    pub_domains = {d for d, n in dom_counts.items() if n >= args.min_domain_count}
    print(f"[overlap] {len(urls)} archive URLs; {len(pub_domains)} publisher domains "
          f"(>= {args.min_domain_count} documents): {', '.join(sorted(pub_domains))}")

    rows = list(csv.DictReader(open(args.list, encoding="utf-8", newline="")))
    results = []
    for r in rows:
        if r["tier"].strip().upper().startswith("X"):
            continue
        base = args.reports / r["folder"] / r["label"]
        h, t = base.with_suffix(".html"), base.with_suffix(".txt")
        if not h.exists() and not t.exists():
            results.append({**r, "status": "not saved", "links": 0, "archive_urls": 0, "publisher_domain_links": 0,
                            "own_site_is_publisher": "", "matched_domains": ""})
            continue
        links = links_of(h if h.exists() else None, t)
        own = domain(r["url"])
        external = {u for u in links if domain(u) != own}
        exact = sum(1 for u in external if norm(u) in urls)
        matched = Counter(domain(u) for u in external if domain(u) in pub_domains)
        results.append({**r, "status": "checked", "links": len(external), "archive_urls": exact,
                        "publisher_domain_links": sum(matched.values()),
                        "own_site_is_publisher": "YES" if own in pub_domains else "no",
                        "matched_domains": "; ".join(f"{d}:{n}" for d, n in matched.most_common(5))})

    fields = ["folder", "country", "years", "tier", "label", "status", "links", "archive_urls",
              "publisher_domain_links", "own_site_is_publisher", "matched_domains"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"{'label':45s} {'tier':4s} {'links':>5s} {'exact':>5s} {'pubdom':>6s} own  matched")
    for x in results:
        if x["status"] != "checked":
            print(f"{x['label']:45s} {x['tier']:4s}  (not saved)")
            continue
        flag = " <-- look" if (x["archive_urls"] >= 3 or x["own_site_is_publisher"] == "YES") else ""
        print(f"{x['label']:45s} {x['tier']:4s} {x['links']:5d} {x['archive_urls']:5d} {x['publisher_domain_links']:6d} "
              f"{x['own_site_is_publisher']:4s} {x['matched_domains']}{flag}")
    print(f"[overlap] written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
