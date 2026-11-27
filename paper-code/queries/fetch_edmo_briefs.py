"""
fetch_edmo_briefs.py -- download the EDMO monthly fact-checking briefs
(no. 1, July 2021, to no. 60, June 2026), the "Monthly Reports" that replaced
them in September 2026, and the special edition, as PDFs, with an index.

The briefs are the tier-1 report source for the EU-related cells of the query
set (United Kingdom, Ireland, France): every
month EDMO asks the fact-checkers of its network which disinformation
narratives circulated most and writes them up by topic and by country.

How the site is organised (checked 2026-10-08): the listing page
https://edmo.eu/fact-checking-briefs/ is built by JavaScript and shows no
links, but the WordPress sitemap https://edmo.eu/publications-sitemap.xml
lists one page per brief, https://edmo.eu/publications/monthly-fact-checking-
brief-no-N-is-out/, and each page links two PDFs of the same brief, a
"Vertical" (portrait) and a "Horizontal" (landscape) layout, under
https://edmo.eu/wp-content/uploads/. The site answers 403 to a non-browser
User-Agent, so the requests carry a browser-like one; robots.txt allows
everything. One request every --delay seconds.

Usage:
  python queries/fetch_edmo_briefs.py --out reports/edmo-briefs [--layout vertical|horizontal|all]
                                      [--limit N] [--delay 1.5] [--text] [--dry-run]

Output: OUT/brief-NN/<file>.pdf (plus .txt with --text, if pdftotext is on
the PATH), OUT/index.csv and OUT/index.md (number, date, title, the month's
focus, the PDF URL, the local file, bytes, sha256, status). Re-running skips
the files already there. Not covered by the toy test (it needs the network).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_pages import pdf_to_text, find_pdftotext, record_version  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SITEMAP = "https://edmo.eu/publications-sitemap.xml"
# The site's firewall answers 403 to anything but a plain browser string (tested
# 2026-10-08: appending a project tag to it is refused), so the script cannot
# identify itself in the User-Agent; it is polite through --delay instead.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")

PAGE_PATTERNS = [
    ("brief", re.compile(r"/publications/monthly-fact-checking-brief-no-(\d+)-is-out(?:-\d+)?/?$")),
    ("report", re.compile(r"/publications/monthly-report-no-(\d+)-is-out/?$")),
    ("special", re.compile(r"/publications/special-edition-brief-is-out/?$")),
]
KIND_ORDER = {"brief": 0, "special": 1, "report": 2}


def get(url: str, delay: float, retries: int = 3) -> bytes:
    """One GET with a browser-like User-Agent, a pause before it, and retries."""
    last: Exception | None = None
    for attempt in range(retries):
        time.sleep(delay * (attempt + 1))
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last = exc
    raise RuntimeError(f"GET failed after {retries} attempts: {url}: {last}")


def list_pages(sitemap_xml: str) -> list[dict]:
    """The brief pages in the sitemap, as dicts (kind, number, url), sorted."""
    pages = []
    for loc in re.findall(r"<loc>(.*?)</loc>", sitemap_xml):
        url = html.unescape(loc.strip())
        for kind, pat in PAGE_PATTERNS:
            m = pat.search(url)
            if m:
                number = int(m.group(1)) if m.groups() else 0
                pages.append({"kind": kind, "number": number, "url": url})
                break
    pages.sort(key=lambda p: (KIND_ORDER[p["kind"]], p["number"]))
    return pages


def layout_of(pdf_url: str) -> str:
    """'vertical', 'horizontal' or 'unknown' from the file name (EDMO names them
    Vertical/Horizontal, or _V / _H; brief no. 1 uses a Greek capital eta for H)."""
    name = urllib.parse.unquote(pdf_url.rsplit("/", 1)[-1])
    stem = re.sub(r"\.pdf$", "", name, flags=re.I)
    # tolerant on purpose: EDMO wrote "Verical" (no. 27) and "_V-" (no. 7)
    if re.search(r"ver[a-z]*cal|(^|[^A-Za-z])V[^A-Za-z]*$", stem, flags=re.I):
        return "vertical"
    if re.search(r"hori[a-z]*tal|(^|[^A-Za-z])[HΗ][^A-Za-z]*$", stem, flags=re.I):
        return "horizontal"
    return "unknown"


def parse_page(page_html: str) -> dict:
    """Title, published date, the one-line focus (og:description) and the PDF links of a brief page."""
    def meta(prop: str) -> str:
        m = re.search(r'<meta\s+property="%s"\s+content="([^"]*)"' % re.escape(prop), page_html)
        return html.unescape(m.group(1)).strip() if m else ""
    title = re.search(r"<title>(.*?)</title>", page_html, flags=re.S)
    date = meta("article:published_time")
    if not date:  # the brief pages carry the date only in their JSON-LD block
        m = re.search(r'"datePublished":"([^"]+)"', page_html)
        date = m.group(1) if m else ""
    pdfs: list[str] = []
    for href in re.findall(r'href="(https://edmo\.eu/wp-content/uploads/[^"]+\.pdf)"', page_html, flags=re.I):
        href = html.unescape(href)
        if href not in pdfs:
            pdfs.append(href)
    return {
        "title": html.unescape(title.group(1)).replace(" - EDMO", "").strip() if title else "",
        "date": date[:10],
        "focus": meta("og:description"),
        "pdfs": pdfs,
    }


def choose(pdfs: list[str], layout: str) -> list[str]:
    """The PDFs to download: the requested layout, or all of them if none has it."""
    if layout == "all":
        return pdfs
    wanted = [p for p in pdfs if layout_of(p) == layout]
    return wanted if wanted else pdfs


def safe_name(pdf_url: str) -> str:
    name = urllib.parse.unquote(pdf_url.rsplit("/", 1)[-1])
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "edmo-briefs",
                    help="folder for the PDFs and the index (default reports/edmo-briefs at the repo root)")
    ap.add_argument("--layout", choices=["vertical", "horizontal", "all"], default="vertical",
                    help="which layout of each brief to download (both hold the same text; default vertical)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N briefs (0 = all)")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    ap.add_argument("--text", action="store_true", help="also extract the text of every PDF (fetch_pages.py's rule)")
    ap.add_argument("--retext", action="store_true", help="with --text: re-extract texts that already exist")
    ap.add_argument("--dry-run", action="store_true", help="list the pages and PDFs, download nothing")
    args = ap.parse_args()

    print(f"[fetch] sitemap {SITEMAP}")
    pages = list_pages(get(SITEMAP, args.delay).decode("utf-8", "replace"))
    print(f"[fetch] {len(pages)} brief pages in the sitemap "
          f"({sum(p['kind'] == 'brief' for p in pages)} monthly briefs, "
          f"{sum(p['kind'] == 'report' for p in pages)} monthly reports, "
          f"{sum(p['kind'] == 'special' for p in pages)} special)")
    if args.limit:
        pages = pages[: args.limit]

    pdftotext = find_pdftotext() if args.text else None
    if args.text and not pdftotext:
        print("[fetch] WARNING: --text asked but pdftotext is not on the PATH; no .txt will be written")

    rows: list[dict] = []
    for p in pages:
        label = f"{p['kind']}-{p['number']:02d}"
        try:
            info = parse_page(get(p["url"], args.delay).decode("utf-8", "replace"))
        except RuntimeError as exc:
            print(f"[fetch] {label}: page failed: {exc}")
            rows.append({**p, "label": label, "status": "page-failed"})
            continue
        wanted = choose(info["pdfs"], args.layout)
        print(f"[fetch] {label} {info['date']} {info['title']!r}: {len(info['pdfs'])} PDF links, taking {len(wanted)}")
        if not wanted:
            rows.append({**p, **info, "label": label, "pdf_url": "", "status": "no-pdf"})
            continue
        for pdf_url in wanted:
            row = {**p, "label": label, "title": info["title"], "date": info["date"], "focus": info["focus"],
                   "pdf_url": pdf_url, "layout": layout_of(pdf_url), "local": "", "bytes": 0, "sha256": "", "status": ""}
            dest = args.out / label / safe_name(pdf_url)
            if args.dry_run:
                row["status"] = "dry-run"
                rows.append(row)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size > 0:
                row["status"] = "already-there"
            else:
                try:
                    data = get(pdf_url, args.delay)
                except RuntimeError as exc:
                    print(f"[fetch]   {pdf_url}: {exc}")
                    row["status"] = "download-failed"
                    rows.append(row)
                    continue
                if not data.startswith(b"%PDF"):
                    print(f"[fetch]   {pdf_url}: not a PDF ({len(data)} bytes), skipped")
                    row["status"] = "not-pdf"
                    rows.append(row)
                    continue
                dest.write_bytes(data)
                row["status"] = "downloaded"
            row["local"] = str(dest.relative_to(args.out)).replace("\\", "/")
            row["bytes"] = dest.stat().st_size
            row["sha256"] = sha256_of(dest)
            if pdftotext:
                txt = dest.with_suffix(".txt")
                if not txt.exists() or args.retext:
                    pdf_to_text(pdftotext, dest, txt)  # the rule of fetch_pages.py: reading order for two-column pages
            rows.append(row)

    if args.dry_run:
        for r in rows:
            print(f"  {r['label']}  {r.get('date','')}  {r.get('pdf_url','')}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    if pdftotext:
        record_version(args.out.parent, pdftotext)
    fields = ["label", "kind", "number", "date", "title", "focus", "url", "pdf_url", "layout", "local", "bytes", "sha256", "status"]
    with open(args.out / "index.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    with open(args.out / "index.md", "w", encoding="utf-8") as f:
        f.write(f"# EDMO briefs downloaded by paper-code/queries/fetch_edmo_briefs.py on {time.strftime('%Y-%m-%d')}\n\n")
        f.write("Source: https://edmo.eu/fact-checking-briefs/ (pages found through the publications sitemap). "
                "Each brief lists the disinformation narratives of the previous month, by topic and by country, "
                "as reported by the fact-checkers of the EDMO network; it is a digest of their fact-checks "
                "(mostly in languages other than English), which the paper discloses.\n\n")
        f.write("| label | date | title | focus | file | status |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['label']} | {r.get('date','')} | {r.get('title','')} | {r.get('focus','')} | "
                    f"{r.get('local','')} | {r.get('status','')} |\n")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"[fetch] done: {counts}; index in {args.out / 'index.csv'}")
    return 0 if all(r["status"] in ("downloaded", "already-there") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
