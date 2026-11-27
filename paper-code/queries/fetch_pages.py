"""
fetch_pages.py -- save the report sources of the query collection (web pages
and PDFs listed in queries/report_sources.csv) into reports/<folder>/, with the
text extracted and an index, so that nothing is saved by hand and the
released source log can cite the exact document (URL, date fetched, sha256).

The list is the source of truth: one row per document with the country, the
years it covers, the tier (1 = monitoring body, official register, research or
parliamentary report; 2 = a review by a fact-checker that is in the archive),
a short label used as the file name, the URL and a note. Add a row, run again:
documents already saved are skipped.

Usage:
  python queries/fetch_pages.py [--list queries/report_sources.csv] [--out reports]
                                [--only LABEL[,LABEL]] [--delay 1.5] [--dry-run] [--retext]

Output: reports/<folder>/<label>.pdf or .html, plus <label>.txt (pdftotext for
PDFs when it is on the PATH, with -layout unless more than 20 percent of the
lines are two-column, then in reading order; a plain-text rendering for HTML;
--retext re-extracts the PDFs already saved), and
reports/sources-index.csv (every row of the list with status, bytes, sha256,
content type, date fetched). Not covered by the toy test (network).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

try:  # Python on Windows may lack the system roots some sites need (arxiv.org failed); certifi has them
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover
    SSL_CONTEXT = ssl.create_default_context()

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
# A plain browser string: several of the sites refuse anything else (see fetch_edmo_briefs.py).
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
FIELDS = ["folder", "country", "years", "tier", "label", "url", "note"]
INDEX_FIELDS = FIELDS + ["status", "content_type", "local", "bytes", "sha256", "fetched"]


def get(url: str, delay: float, retries: int = 3, insecure: bool = False) -> tuple[bytes, str]:
    """GET with a browser-like User-Agent and a pause; returns (body, content type).
    insecure=True skips certificate checks (for a government site with a broken chain); the index records it."""
    last: Exception | None = None
    ctx = SSL_CONTEXT
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    for attempt in range(retries):
        time.sleep(delay * (attempt + 1))
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8",
                                                   "Accept-Language": "en-US,en;q=0.9"})
        try:
            with urllib.request.urlopen(req, timeout=90, context=ctx) as resp:
                return resp.read(), resp.headers.get("Content-Type", "")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last = exc
    raise RuntimeError(f"GET failed after {retries} attempts: {last}")


class TextOf(HTMLParser):
    """A small HTML-to-text: drops script, style and the like, keeps one line per block element."""
    DROP = {"script", "style", "noscript", "svg", "template", "iframe"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section",
             "article", "header", "footer", "blockquote", "figcaption", "td", "th", "ul", "ol", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.DROP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.DROP and self.skip:
            self.skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.split("\n")]
        out, blank = [], 0
        for line in lines:
            blank = blank + 1 if not line else 0
            if blank <= 1:
                out.append(line)
        return "\n".join(out).strip() + "\n"


def html_title(page: bytes) -> str:
    m = re.search(rb"<title[^>]*>(.*?)</title>", page, flags=re.S | re.I)
    return html.unescape(m.group(1).decode("utf-8", "replace")).strip() if m else ""


TWO_COLUMN_SHARE = 0.20  # above this share of two-column lines, the text is taken in reading order


def two_column_share(text: str) -> float:
    """Share of non-empty lines that hold two pieces of text separated by six or more spaces: what pdftotext -layout
    writes when a page has two columns side by side (each printed line then mixes the two columns)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return sum(1 for l in lines if re.search(r"\S {6,}\S", l)) / max(1, len(lines))


VERSION_FILE = "pdftotext-version.txt"  # beside the texts: which pdftotext made them


def find_pdftotext() -> str | None:
    """The one pdftotext every step uses: the PDFTOTEXT environment variable if it is set, else the first pdftotext on
    the PATH. Several can be installed (for example Poppler 25.07 from winget first on the PowerShell PATH,
    MiKTeX's Poppler 24.04, and xpdf 4.00 inside Git Bash), and they write slightly different text."""
    import os
    exe = os.environ.get("PDFTOTEXT") or shutil.which("pdftotext")
    if exe and not Path(exe).exists():
        exe = shutil.which(exe)
    return exe or None


def pdftotext_version(exe: str) -> str:
    """The first line of 'pdftotext -v', e.g. 'pdftotext version 25.07.0'."""
    r = subprocess.run([exe, "-v"], capture_output=True, text=True)
    return ((r.stderr or r.stdout).strip().splitlines() or [""])[0].strip()


def record_version(reports: Path, exe: str) -> None:
    """Write the version beside the texts; warn when texts of another version are already there."""
    v = pdftotext_version(exe)
    f = reports / VERSION_FILE
    old = f.read_text(encoding="utf-8").strip() if f.exists() else ""
    if old and old != v:
        print(f"[fetch] WARNING: the saved texts were made by '{old}', this one is '{v}': run --retext so that all come from one version")
    f.write_text(v + "\n", encoding="utf-8")


def pdf_to_text(pdftotext: str, pdf: Path, txt: Path) -> str:
    """Extract a PDF's text. First with -layout (keeps tables and line structure); if more than TWO_COLUMN_SHARE of
    the lines are two-column, extract again in reading order (no -layout), because a sentence split across the
    interleaved columns cannot be quoted verbatim (found 2026-10-15: Hope not Hate 2022, 9 of 67 drafted passages
    found in the layout text, 56 in the reading-order text). Returns the mode used: "layout" or "reading-order"."""
    subprocess.run([pdftotext, "-layout", str(pdf), str(txt)], check=False)
    if not txt.exists():
        return ""
    if two_column_share(txt.read_text(encoding="utf-8", errors="replace")) > TWO_COLUMN_SHARE:
        subprocess.run([pdftotext, str(pdf), str(txt)], check=False)
        return "reading-order"
    return "layout"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", type=Path, default=HERE / "report_sources.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "reports")
    ap.add_argument("--only", default="", help="comma-separated labels to fetch (default all)")
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--insecure", default="", help="comma-separated labels fetched without certificate checks (recorded in the index)")
    ap.add_argument("--retext", action="store_true", help="re-extract the text of every PDF already saved (the EDMO briefs too), with the extraction rule; download nothing")
    args = ap.parse_args()
    insecure = {s.strip() for s in args.insecure.split(",") if s.strip()}

    with open(args.list, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("url", "").strip() and not r["url"].startswith("#")]
    missing = [k for k in FIELDS if rows and k not in rows[0]]
    if missing:
        print(f"[fetch] the list lacks the columns {missing}")
        return 2
    labels = {r["label"] for r in rows}
    if len(labels) != len(rows):
        dup = sorted({r["label"] for r in rows if sum(x["label"] == r["label"] for x in rows) > 1})
        print(f"[fetch] duplicate labels in the list: {dup}")
        return 2
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if only:
        rows = [r for r in rows if r["label"] in only]
    pdftotext = find_pdftotext()
    if not pdftotext:
        print("[fetch] WARNING: pdftotext not on the PATH; PDFs are saved without a .txt")
    else:
        print(f"[fetch] {pdftotext_version(pdftotext)}: {pdftotext}")

    index: list[dict] = []
    today = time.strftime("%Y-%m-%d")
    for r in rows:
        folder = args.out / r["folder"]
        base = folder / r["label"]
        rec = {**r, "status": "", "content_type": "", "local": "", "bytes": 0, "sha256": "", "fetched": ""}
        if r["tier"].strip().upper().startswith("X"):
            # an excluded source: kept in the list so the exclusion is on record,
            # never fetched, and any file fetched before the exclusion is removed
            for p in (base.with_suffix(".pdf"), base.with_suffix(".html"), base.with_suffix(".txt")):
                if p.exists():
                    p.unlink()
            rec["status"] = "excluded"
            index.append(rec)
            continue
        existing = [p for p in (base.with_suffix(".pdf"), base.with_suffix(".html")) if p.exists() and p.stat().st_size > 0]
        if existing:
            dest = existing[0]
            if args.retext and dest.suffix == ".pdf" and pdftotext:
                mode = pdf_to_text(pdftotext, dest, base.with_suffix(".txt"))
                if mode == "reading-order":
                    print(f"[fetch] {r['label']:40s} text re-extracted in reading order (two-column layout)")
            rec.update(status="already-there", local=str(dest.relative_to(args.out)).replace("\\", "/"),
                       bytes=dest.stat().st_size, sha256=sha256_of(dest))
            index.append(rec)
            continue
        if args.dry_run:
            print(f"[fetch] would fetch {r['label']:40s} {r['url']}")
            rec["status"] = "dry-run"
            index.append(rec)
            continue
        try:
            body, ctype = get(r["url"], args.delay, insecure=r["label"] in insecure)
        except RuntimeError as exc:
            print(f"[fetch] {r['label']}: {exc}")
            rec["status"] = "failed"
            index.append(rec)
            continue
        is_pdf = body.startswith(b"%PDF")
        folder.mkdir(parents=True, exist_ok=True)
        dest = base.with_suffix(".pdf" if is_pdf else ".html")
        dest.write_bytes(body)
        txt = base.with_suffix(".txt")
        if is_pdf:
            if pdftotext:
                pdf_to_text(pdftotext, dest, txt)
        else:
            parser = TextOf()
            parser.feed(body.decode("utf-8", "replace"))
            title = html_title(body)
            txt.write_text((title + "\n" + r["url"] + "\n\n" if title else "") + parser.text(), encoding="utf-8")
        rec.update(status="downloaded" + (" (insecure)" if r["label"] in insecure else ""),
                   content_type=("application/pdf" if is_pdf else ctype.split(";")[0]),
                   local=str(dest.relative_to(args.out)).replace("\\", "/"), bytes=len(body),
                   sha256=sha256_of(dest), fetched=today)
        print(f"[fetch] {r['label']:40s} {rec['status']} {len(body):>9d} bytes {'pdf' if is_pdf else 'html'}")
        index.append(rec)

    if args.retext and pdftotext:
        # the EDMO briefs (fetched by fetch_edmo_briefs.py) are re-extracted with the same program and rule
        n = 0
        for pdf in sorted((args.out / "edmo-briefs").rglob("*.pdf")):
            if pdf.with_suffix(".txt").exists():
                pdf_to_text(pdftotext, pdf, pdf.with_suffix(".txt"))
                n += 1
        print(f"[fetch] {n} EDMO brief texts re-extracted")
    if pdftotext and not args.dry_run:
        record_version(args.out, pdftotext)
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        idx_path = args.out / "sources-index.csv"
        # keep the rows of earlier runs that were not in this run (for --only)
        old: dict[str, dict] = {}
        if idx_path.exists():
            with open(idx_path, encoding="utf-8", newline="") as f:
                old = {x["label"]: x for x in csv.DictReader(f)}
        for rec in index:
            old[rec["label"]] = {k: rec.get(k, "") for k in INDEX_FIELDS}
        try:
            f = open(idx_path, "w", encoding="utf-8", newline="")
        except PermissionError:  # the index is open in Excel: write beside it rather than lose the run
            idx_path = idx_path.with_name("sources-index.new.csv")
            print(f"[fetch] WARNING: sources-index.csv is locked (open in Excel?); writing {idx_path.name} instead")
            f = open(idx_path, "w", encoding="utf-8", newline="")
        with f:
            w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
            w.writeheader()
            for rec in sorted(old.values(), key=lambda x: (x["folder"], x["years"], x["label"])):
                w.writerow(rec)
    counts: dict[str, int] = {}
    for rec in index:
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    print(f"[fetch] done: {counts}")
    return 0 if not counts.get("failed") else 1


if __name__ == "__main__":
    sys.exit(main())
