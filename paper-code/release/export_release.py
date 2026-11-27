"""
export_release.py -- assemble the test collection for Zenodo.

Copies into --out:
  queries.json                    all queries with all formulas (the file given by --queries; version 7), WITHOUT the
                                  report's own words: provenance.passage, provenance.rewrite_context,
                                  provenance.rewrite_closed.context and the rewrite's notes (provenance.rewrite_closed.note,
                                  which sometimes quote them) are removed (no quotation of the
                                  reports is released); the source's address, file name and checksum stay. The query
                                  text stays even where it repeats a claim in the words the report used to report it
  qrels/qrels_ai.txt, qrels_human.txt, qrels_ai_check.txt, pool_depth.jsonl (qrels/build_qrels.py)
  judge/labels_llm_*.jsonl        every AI judge's labels (the main judge's and the others', for the robustness table)
  pool/check.json, pool/shared.json  the human check sample: queries, documents, who judged what
  runs/*.trec, runs/manifest.json (the main text runs) and manifest_claimart.json (the second row), runs/*_sets.jsonl, runs/*_report.md
  documents.jsonl                 doc_id, date, publisher and the claim text cut at --text-chars (300);
                                  no explanation bodies (CimpleKG is CC BY-NC-SA 4.0; we redistribute ids and short claim texts)
  labels/agreement.md, judge/validation.md, eval/*.md  (when present)
  README.md (release/README-DATA.md: what every file holds), MODELS.md, LICENSE, reproduce.sh, CHECKSUMS.sha256
  drafting_model.json (with --drafting DIR, the folder of the drafting run): the model that drafted the candidates, as
                                  every raw reply recorded it: the model name, its precision, the Ollama version, the
                                  model's digest (the checksum of the model file), the pdftotext version and the sha256
                                  of the prompt, with the number of replies; no reply and no text of a report
  writers/ (with --writers DIR, the folder of the writers' sheets):
    wikipedia_redirects.json, wikipedia_incoming_redirects.json, wikipedia_disambiguation.json
                                  Wikipedia's answers the import and the title check used, so that both can be
                                  repeated offline with the same result
    formula_repairs.csv           every formula repaired before the runs (repair_log.csv): query, writer id,
                                  the formula as written and as repaired, the rule, the date, whether the writer confirmed
    title_changes.csv             every formula changed by the writers' choice of page for a flagged title
                                  (titoli_applicati_log.csv): query, writer id, before, after, the changes, the date
    Only these columns are written: no first names, no notes in Italian.
  pool/pool.jsonl                 every judged pair with the rank each system gave it and whether it is in the human
                                  check sample (query_id, doc_id, ranks, min_rank, check); a pair added by
                                  pool/extend_pool_before.py also carries added_for ("before YYYY-MM") and
                                  ranks_before (its ranks in the lists cut to that month); the texts are left out
                                  (Section 5: "we release the pool")
  labels/labels_<judge>.jsonl     the human judges' labels (Section 6: "the human judges' label files"), without the
                                  free note, the timestamp and the seconds; a file kept as *_before_* by
                                  pool/judges_sheet.py read is left out; labels_adjudicated.jsonl, when
                                  judge/adjudicate.py wrote it, holds only the shared documents both judges agree on
                                  (the paper's human label is the owner's, common/labels.final_check_labels)
  snapshot/ (with --snap DIR, the snapshot folder): mirror_manifest.json (the mirror's checksum),
                                  snapshot_manifest.json (the cleaning rules and their counts), language_decisions.jsonl
                                  (every language decision with its measurements, Section 3), ifcn_sources.json (the
                                  IFCN file as downloaded, Section 3)
  sources.csv (with --sources FILE, reports/sources-index.csv): the report list, one line per saved
                                  document: folder, country, years, tier, label, url, content type, bytes, sha256, fetch
                                  date (Section 5: "the released source list"); the free note and the local status are left out
  --humans ID1,ID2               the human judges' ids in the work folder, in the order the paper numbers them: in the
                                  release they are H1 and H2, as in the paper (the label files' names, their annotator
                                  field and the name of the sheet each label came from, check.json, and the three reports
                                  that name them: labels/agreement.md, judge/validation.md, pool/pool_summary.md)
Never released: the drafting files (raw replies, candidates, merged clusters, review log, draw report), which quote
the reports. The writers' instruction is in the code (queries/FORMULA-INSTRUCTION.md), not in this export.
Usage: python release/export_release.py --work WORK --queries Q.json --out release_build [--text-chars 300] [--writers DIR] [--snap DIR]
       [--sources FILE] [--humans ID1,ID2] [--drafting DIR] [--code-url URL]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import kg  # noqa: E402
from common.jsonl import read_jsonl, write_jsonl  # noqa: E402

HERE = Path(__file__).resolve().parent.parent


# the columns released from the two logs (the others hold first names or notes for the writers)
LOGS = {"repair_log.csv": ("formula_repairs.csv", ["query_id", "writer", "original", "repaired", "rule", "date", "confirm_with_writer"]),
        "titoli_applicati_log.csv": ("title_changes.csv", ["query_id", "writer", "formula_before", "formula_after", "changes", "applied_on"])}
CACHES = ("wikipedia_redirects.json", "wikipedia_incoming_redirects.json", "wikipedia_disambiguation.json")
POOL_FIELDS = ("query_id", "doc_id", "ranks", "min_rank", "check")
POOL_FIELDS_IF_ADDED = ("added_for", "ranks_before")  # only on the pairs added by pool/extend_pool_before.py
# the fields that copy a report's words: the passage, the report's words used in the rewrite, and the rewrite's notes
# (11 of which quote those words); the query text itself is released even when it repeats the claim as the report gave it
QUOTATIONS = (("provenance", "passage"), ("provenance", "rewrite_context"), ("provenance", "rewrite_closed", "context"),
              ("provenance", "rewrite_closed", "note"))
SOURCE_FIELDS = ("folder", "country", "years", "tier", "label", "url", "content_type", "bytes", "sha256", "fetched")


def without_quotations(queries: list[dict]) -> tuple[list[dict], int]:
    """The query file without the report's own words (QUOTATIONS); returns the copy and how many fields were removed."""
    n = 0
    out = json.loads(json.dumps(queries))
    for q in out:
        for path in QUOTATIONS:
            d = q
            for key in path[:-1]:
                d = d.get(key) if isinstance(d, dict) else None
            if isinstance(d, dict) and path[-1] in d:
                del d[path[-1]]
                n += 1
    return out, n
LABEL_DROP = ("note", "ts", "seconds")  # a judge's free note, when, how long: not part of the judgment
HUMAN_REPORTS = (("labels", "agreement.md"), ("judge", "validation.md"), ("pool", "pool_summary.md"))


def rename_humans(text: str, names: dict[str, str]) -> str:
    """Replace each judge id that stands alone (not inside a word, a measure name such as R@B or set_R, or an initial
    such as 'R.') by the paper's name."""
    alt = "|".join(re.escape(k) for k in sorted(names, key=len, reverse=True))
    return re.sub(rf"(?<![\w@.])({alt})(?![\w@.])", lambda m: names[m.group(1)], text)


def rename_sheet(name: str, names: dict[str, str]) -> str:
    """The judge's id inside the name of a sheet file: sheet_R.xlsx -> sheet_H1.xlsx, sheet_M_round2.xlsx -> sheet_H2_round2.xlsx."""
    for k in sorted(names, key=len, reverse=True):
        name = re.sub(rf"(?<=_){re.escape(k)}(?=[._])", names[k], name)
    return name


# the fields of a raw reply of queries/draft_needs.py that describe the software, never the reply or the text it read
DRAFTING_FIELDS = ("model", "precision", "ollama_version", "model_digest", "pdftotext", "prompt_sha256", "reasoning",
                   "max_tokens", "frequency_penalty")


def drafting_model(raw_dir: Path) -> dict:
    """The software state of the drafting run, from every raw reply (DIR/raw/<cell>/<file>.json): each distinct
    combination of DRAFTING_FIELDS with its number of replies. A run mixes no two states (draft_needs.py refuses it)."""
    states: dict[str, dict] = {}
    files = sorted(raw_dir.rglob("*.json"))
    if not files:
        raise SystemExit(f"--drafting: no raw reply under {raw_dir}")
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        state = {k: r.get(k) for k in DRAFTING_FIELDS}
        key = json.dumps(state, sort_keys=True)
        states.setdefault(key, {**state, "replies": 0})["replies"] += 1
    return {"what": "the model that drafted the candidate claims (queries/draft_needs.py), as every raw reply of the run "
                    "recorded it; model_digest is Ollama's checksum of the model file, prompt_sha256 the checksum of the prompt "
                    "(queries/prompts/draft_needs.txt) with its fields filled",
            "modelfile": "queries/ollama/Modelfile-qwen3-14b-16k (in the code)", "states": list(states.values())}


SNAP_FILES = (("mirror_manifest.json", "mirror_manifest.json"), ("final/snapshot_manifest.json", "snapshot_manifest.json"),
              ("final/language_decisions.jsonl", "language_decisions.jsonl"), ("ifcn_sources.json", "ifcn_sources.json"))


def export_writers(src: Path, dst: Path) -> None:
    """The writers' side of the release: Wikipedia's answers and the two logs, columns filtered (see the docstring)."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in CACHES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    for name, (new, cols) in LOGS.items():
        if not (src / name).exists():
            continue
        with open(src / name, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        with open(dst / new, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-chars", type=int, default=300)
    ap.add_argument("--writers", default=None, help="the writers' folder: Wikipedia caches, repair_log.csv, titoli_applicati_log.csv")
    ap.add_argument("--snap", default=None, help="the snapshot folder: mirror and snapshot manifests, language decisions, IFCN file")
    ap.add_argument("--sources", default=None, help="reports/sources-index.csv: the report list with addresses, fetch dates and checksums")
    ap.add_argument("--humans", default=None, help="the two human judges' ids, in the paper's order: released as H1 and H2")
    ap.add_argument("--drafting", default=None, help="the folder of the drafting run (with raw/): writes drafting_model.json")
    ap.add_argument("--code-url", default="the code repository named in the paper",
                    help="the address of the code repository, written into the release's README")
    args = ap.parse_args()
    ids = [x.strip() for x in (args.humans or "").split(",") if x.strip()]
    if args.humans and len(ids) != 2:
        raise SystemExit("--humans must name the two human judges")
    humans = {h: f"H{i + 1}" for i, h in enumerate(ids)}
    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    queries, n_removed = without_quotations(json.loads(Path(args.queries).read_text(encoding="utf-8")))
    (out / "queries.json").write_text(json.dumps(queries, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"queries.json: {len(queries)} queries, {n_removed} quotation fields of the reports removed")
    for sub, pattern in (("qrels", "*"), ("runs", "*.trec"), ("runs", "manifest*.json"), ("runs", "*_sets.jsonl"), ("runs", "*_report.md"),
                         ("labels", "agreement.md"), ("judge", "validation.md"), ("judge", "labels_llm_*.jsonl"), ("eval", "*.md"), ("stats", "*.md"),
                         ("pool", "pool_summary.md"), ("pool", "check.json"), ("pool", "shared.json")):
        src = work / sub
        if src.exists():
            (out / sub).mkdir(exist_ok=True)
            for p in src.glob(pattern):
                if p.is_file():
                    shutil.copy2(p, out / sub / p.name)
    n = write_jsonl(out / "documents.jsonl", ({"doc_id": d["cr_iri"], "date_published": d.get("date_published"), "publisher": d.get("author_name"),
                                                "claim_text": d["claim_text"][:args.text_chars]} for d in kg.load_corpus(work / "corpus.jsonl")))
    if args.writers:
        export_writers(Path(args.writers), out / "writers")
    if (work / "pool" / "pool.jsonl").exists():
        write_jsonl(out / "pool" / "pool.jsonl", ({**{k: r.get(k) for k in POOL_FIELDS}, **{k: r[k] for k in POOL_FIELDS_IF_ADDED if k in r}}
                                                  for r in read_jsonl(work / "pool" / "pool.jsonl")))
    for p in sorted((work / "labels").glob("labels_*.jsonl")) if (work / "labels").exists() else []:
        if "_before_" in p.name:
            continue
        (out / "labels").mkdir(exist_ok=True)
        who = p.stem[len("labels_"):]
        name = f"labels_{humans[who]}.jsonl" if who in humans else p.name
        write_jsonl(out / "labels" / name, ({k: (humans.get(v, v) if k == "annotator" else rename_sheet(v, humans) if k == "sheet" and isinstance(v, str)
                                                 else v) for k, v in r.items() if k not in LABEL_DROP}
                                            for r in read_jsonl(p)))
    if humans:
        chk = out / "pool" / "check.json"
        if chk.exists():
            c = json.loads(chk.read_text(encoding="utf-8"))
            c["judges"] = [humans.get(j, j) for j in c.get("judges", [])]
            for pq in (c.get("per_query") or {}).values():
                if "owner" in pq:
                    pq["owner"] = humans.get(pq["owner"], pq["owner"])
            chk.write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        for sub, name in HUMAN_REPORTS:
            f = out / sub / name
            if f.exists():   # bytes in and out: the line endings stay as they were
                f.write_bytes(rename_humans(f.read_bytes().decode("utf-8"), humans).encode("utf-8"))
    if args.sources:
        with open(args.sources, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        with open(out / "sources.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SOURCE_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    if args.snap:
        snap = Path(args.snap)
        (out / "snapshot").mkdir(exist_ok=True)
        for src, name in SNAP_FILES:
            if not (snap / src).exists():
                raise SystemExit(f"--snap: missing {snap / src}")
            shutil.copy2(snap / src, out / "snapshot" / name)
    if args.drafting:
        (out / "drafting_model.json").write_text(json.dumps(drafting_model(Path(args.drafting) / "raw"), indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8", newline="\n")
    for name in ("MODELS.md", "LICENSE", "reproduce.sh"):
        if (HERE / name).exists():
            shutil.copy2(HERE / name, out / name)
    # the README of the collection (what every file holds), not the README of the code; @CODE_URL@ becomes --code-url
    readme = (HERE / "release" / "README-DATA.md").read_bytes().replace(b"\r\n", b"\n")
    (out / "README.md").write_bytes(readme.replace(b"@CODE_URL@", args.code_url.encode("utf-8")))
    lines = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "CHECKSUMS.sha256":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}")
    # Unix line endings on every system: sha256sum -c would read a CR as the last letter of the file name
    (out / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"release folder {out}: {len(lines)} files, {n} documents")


if __name__ == "__main__":
    main()
