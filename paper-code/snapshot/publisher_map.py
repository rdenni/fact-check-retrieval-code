"""
publisher_map.py -- build the publisher -> country / IFCN-status table from
the store's publishers and the IFCN signatory file of the claimreview-data
project (the same source CimpleKG's builders use, CimpleKG paper step 3).

Where the IFCN file comes from: https://github.com/MartinoMensio/claimreview-data/releases,
the latest release's zip (about 340 MB), member <date>/ifcn_sources.json
(0.2 MB): 226 signatories on 2026-09-24, each with original.name, country,
language, website, issued_on, expired, and original_label 'Valid IFCN
signatory' or 'Expired IFCN signatory'.

Matching: by web domain. The store's publisher node has a url
(http://www.snopes.com); the IFCN entry has a domain (snopes.com). Both are
reduced to a bare host without 'www.' and compared; a publisher host that
ends with an IFCN domain (fact-check.example.org vs example.org) also matches.
Rows that do not match are left with empty country and status, to be filled
by hand (the CimpleKG paper's Table 4 and the publishers' websites help).

Output CSV columns: publisher, url, fact_checks, ifcn_name, country, language,
ifcn_status (verified / expired / empty), issued_on, matched_by.
This CSV is the --country-map of stats/archive_stats.py and the
--publisher-map of snapshot/build_store.py.

Usage:
  python snapshot/publisher_map.py --work WORK --ifcn-sources ifcn_sources.json --out WORK/stats/publishers_map.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import kg  # noqa: E402


def host_of(url: str) -> str:
    u = (url or "").strip().lower()
    if not u:
        return ""
    if "://" not in u:
        u = "http://" + u
    h = urlparse(u).netloc.split(":")[0]
    return h[4:] if h.startswith("www.") else h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="needs WORK/corpus.jsonl (run_all.py --steps prepare)")
    ap.add_argument("--store", default=None, help="the store, to read the publishers' urls (optional; names only without it)")
    ap.add_argument("--ifcn-sources", required=True)
    ap.add_argument("--out", default=None, help="default WORK/stats/publishers_map.csv")
    args = ap.parse_args()
    work = Path(args.work)
    out = Path(args.out) if args.out else work / "stats" / "publishers_map.csv"
    corpus = kg.load_corpus(work / "corpus.jsonl")
    counts = Counter(d.get("author_name") or "missing" for d in corpus)
    urls: dict[str, str] = {}
    if args.store:
        store = kg.open_store(args.store)
        q = f'SELECT DISTINCT ?name ?url WHERE {{ ?o <{kg.P_NAME}> ?name . OPTIONAL {{ ?o <{kg.P_URL}> ?url }} }}'
        for row in store.query(q):
            if row["url"]:
                urls.setdefault(row["name"].value, row["url"].value)
    ifcn = json.load(open(args.ifcn_sources, encoding="utf-8"))
    by_domain = {}
    for it in ifcn:
        o = it.get("original") or {}
        dom = host_of(it.get("domain") or o.get("website") or "")
        if dom:
            by_domain[dom] = {"ifcn_name": o.get("name") or it.get("domain"), "country": o.get("country") or "", "language": o.get("language") or "",
                              "ifcn_status": "expired" if (o.get("expired") or "expired" in (it.get("original_label") or "").lower()) else "verified",
                              "issued_on": (o.get("issued_on") or "")[:10]}
    rows, matched = [], 0
    for name, n in counts.most_common():
        host = host_of(urls.get(name, ""))
        hit, how = None, ""
        if host in by_domain:
            hit, how = by_domain[host], "domain"
        else:
            for dom, entry in by_domain.items():
                if host and (host.endswith("." + dom) or dom.endswith("." + host)):
                    hit, how = entry, "parent-domain"
                    break
        if hit:
            matched += 1
        rows.append({"publisher": name, "url": urls.get(name, ""), "fact_checks": n, "ifcn_name": hit["ifcn_name"] if hit else "",
                     "country": hit["country"] if hit else "", "language": hit["language"] if hit else "",
                     "ifcn_status": hit["ifcn_status"] if hit else "", "issued_on": hit["issued_on"] if hit else "", "matched_by": how})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    unmatched = [r for r in rows if not r["country"]]
    print(f"publishers: {len(rows)}; matched to IFCN by domain: {matched}; to fill by hand: {len(unmatched)} "
          f"(covering {sum(r['fact_checks'] for r in unmatched):,} fact-checks) -> {out}")
    for r in unmatched[:15]:
        print(f"  {r['fact_checks']:7d}  {r['publisher'][:40]:40s} {r['url'][:50]}")


if __name__ == "__main__":
    main()
