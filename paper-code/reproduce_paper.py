"""
reproduce_paper.py -- rebuild every number, table and figure of the paper from the
snapshot, in a separate folder, and say for each file whether it is identical to
the one the paper uses (the folder of the paper's generated files, --paper-figures).

Rule: every number, statistic, plot and table of the paper
must be reproduced by a script; what no script reproduces does not go in the paper.
This script is the test of that rule. It runs, in order:

  1  run_all.py --steps prepare --node claim   the documents and the claim-level entities, from the store
  2  run_all.py --steps prepare --node cr      the explanation-level entities, from the store
  3  snapshot/publisher_map.py                 every publisher -> its IFCN country
  4  stats/archive_stats.py                    the counts of the archive
  5  stats/plot_archive.py                     the six figures of Section 3 and numbers.tex
  6  queries/allocate_queries.py               the query allocation (allocation.tex, allocation_numbers.tex, allocation.json)
  7  stats/paper_facts.py                      every other own number and name list (facts_numbers.tex) + the claim checks;
     it reads the allocation of step 6 to count the queries that sit in another year than the allocation planned
  8  model_selection/pareto_simple.py          the model rule (model_numbers.tex)
  9  queries/prepare_cells.py                  the report sources (sources_numbers.tex); needs the folder of the fetched
     reports (queries/fetch_pages.py, queries/fetch_edmo_briefs.py; default reports/ next to this folder, --reports DIR);
     --skip-sources leaves it out
 10  stats/live_checks.py                      only with --live: asks the public endpoint today (live_numbers.tex)
 11  systems/graph_boolean.py + eval/writer_agreement.py --tex   the writers' numbers (writer_numbers.tex): both
     writers' graph runs remade from the store's entity index and the formula file, then their statistics and agreement
     (when the formula file exists); each formula is also run with SPARQL on the store, and the
     claim of Section 4 that the two ways give identical sets is checked query by query
 12  eval/paper_results.py                     every number, table and figure of the Results section (results_*),
     from the result files that the evaluation scripts saved in the work folder of the runs
     (SNAP/runs-main/eval, judge, labels, pool, qrels, runs); it does not re-run the judges or the measures

It also copies queries/prompts/draft_needs.txt to figures/draft_needs_prompt.txt and queries/FORMULA-INSTRUCTION.md, from
its first numbered heading on, to figures/formula_instruction.txt (the Supplementary Material prints both).

Then it compares every file it made with the file of the same name in the folder of the paper's generated files
(--paper-figures; by default latex/figures next to this folder, when it exists): IDENTICAL, DIFFERENT (with the first
lines that differ), or MISSING (the paper uses a file that no step made: that file is not reproduced).
live_numbers.tex is dated, so a difference there is expected and reported as such. Without that folder the files are
remade and not compared. When the paper's LaTeX sources are next to this folder (latex/check_tex.py), it finally
checks that every macro the LaTeX uses is defined by a generated file and lists the numbers typed by hand.

Nothing in the paper's folder changes unless --write is given: then the files made are
copied into the paper's folder and the paper is compiled with pdflatex and bibtex.

The design constants of the allocation (200 queries, floor 3, since 2016, at least 50
documents, Ghana capped at 2) are the paper's design, decided before the draw; they are
written below, not chosen by this script.

Usage (every path can be given):
  python reproduce_paper.py --snap C:\\snapshot-2026-10 [--paper-figures DIR] [--reports DIR] [--live] [--write] [--skip-sources] [--out DIR]
Exit code 0 when every step ran, every file compared is IDENTICAL (live_numbers.tex excepted) and every claim check passes.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_FIGURES = REPO / "latex" / "figures"   # the paper's generated files, when the paper's sources are next to this folder
CHECK_TEX = REPO / "latex" / "check_tex.py"
PY = sys.executable
ALLOCATION = ["--total", "200", "--floor", "3", "--since", "2016", "--min-docs", "50", "--cap", "Ghana=2"]
DATED = {"live_numbers.tex"}  # files that carry the date they were made; a difference is expected


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_differences(a: Path, b: Path, n: int = 3) -> list[str]:
    if a.suffix in (".pdf",):
        return ["(a figure: compare the numbers behind it in figures.json)"]
    la = a.read_text(encoding="utf-8", errors="replace").splitlines()
    lb = b.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for i in range(max(len(la), len(lb))):
        x = la[i] if i < len(la) else "<no line>"
        y = lb[i] if i < len(lb) else "<no line>"
        if x != y:
            out.append(f"line {i + 1}: paper has  {y[:150]}")
            out.append(f"         remade has {x[:150]}")
            if len(out) >= 2 * n:
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snap", required=True, help="the snapshot folder, e.g. C:\\snapshot-2026-10")
    ap.add_argument("--store", default=None, help="default SNAP/final/store")
    ap.add_argument("--mirror", default=None, help="default SNAP/mirror.nt")
    ap.add_argument("--ifcn", default=None, help="default SNAP/ifcn_sources.json")
    ap.add_argument("--runs", default=None, help="manifest.json of the text run; default SNAP/runs-main/runs/manifest.json")
    ap.add_argument("--queries", default=None, help="default SNAP/drafting-work/queries_v7.json")
    ap.add_argument("--pilot", default=None, help="default SNAP/drafting-work/queries_pilot_v7.json")
    ap.add_argument("--formulas", default=None, help="the query file with both writers' formulas; default SNAP/drafting-work/queries_with_formulas.json")
    ap.add_argument("--writers", default=None, help="the writers' folder (repair logs); default SNAP/drafting-work/writers")
    ap.add_argument("--results-work", default=None, help="the work folder of the runs, with the saved result files; default SNAP/runs-main")
    ap.add_argument("--out", default=None, help="where to rebuild; default SNAP/reproduce (never inside the repository)")
    ap.add_argument("--live", action="store_true", help="also ask the public endpoint (step 10)")
    ap.add_argument("--skip-sources", action="store_true", help="skip step 9 (when the reports/ folder is not on this machine)")
    ap.add_argument("--reports", default=None, help="the folder of the fetched reports for step 9; default reports/ next to this folder")
    ap.add_argument("--paper-figures", default=None, help="the folder of the paper's generated files to compare with; "
                    "default latex/figures next to this folder, when it exists")
    ap.add_argument("--write", action="store_true", help="copy the remade files into the paper's folder and compile the paper")
    args = ap.parse_args()
    figures = Path(args.paper_figures) if args.paper_figures else (DEFAULT_FIGURES if DEFAULT_FIGURES.is_dir() else None)
    if figures is not None and not figures.is_dir():
        raise SystemExit(f"--paper-figures: {figures} is not a folder")
    if args.write and figures is None:
        raise SystemExit("--write needs the paper's folder (--paper-figures)")
    snap = Path(args.snap)
    store = Path(args.store or snap / "final" / "store")
    mirror = Path(args.mirror or snap / "mirror.nt")
    ifcn = Path(args.ifcn or snap / "ifcn_sources.json")
    runs = Path(args.runs or snap / "runs-main" / "runs" / "manifest.json")
    queries = Path(args.queries or snap / "drafting-work" / "queries_v7.json")
    pilot = Path(args.pilot or snap / "drafting-work" / "queries_pilot_v7.json")
    out = Path(args.out or snap / "reproduce")
    for p in (store, mirror, ifcn, runs, queries, pilot, snap / "mirror_manifest.json", store.parent / "snapshot_manifest.json"):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")
    if out.resolve().is_relative_to(REPO.resolve()):
        raise SystemExit("--out must be outside the repository (data never goes in the repository)")
    if out.exists():
        shutil.rmtree(out)
    work, figs = out / "work", out / "figures"
    figs.mkdir(parents=True)
    log = []

    def step(k, what, cmd, required=True):
        print(f"\n=== STEP {k}: {what}\n    {' '.join(str(c) for c in cmd)}", flush=True)
        t0 = time.time()
        r = subprocess.run([str(c) for c in cmd], cwd=HERE)
        took = time.time() - t0
        log.append((k, what, r.returncode, took))
        print(f"    -> exit {r.returncode}, {took:.0f} s", flush=True)
        if r.returncode != 0 and required:
            raise SystemExit(f"STEP {k} FAILED ({what}); nothing was changed in the paper's folder")
        return r.returncode

    step(1, "documents and claim-level entities from the store",
         [PY, "run_all.py", "--steps", "prepare", "--node", "claim", "--store", store, "--work", work, "--queries", queries, "--profile", "cpu-test"])
    step(2, "explanation-level entities from the store",
         [PY, "run_all.py", "--steps", "prepare", "--node", "cr", "--store", store, "--work", work, "--queries", queries, "--profile", "cpu-test"])
    step(3, "publisher -> IFCN country", [PY, "snapshot/publisher_map.py", "--work", work, "--store", store, "--ifcn-sources", ifcn])
    step(4, "the counts of the archive", [PY, "stats/archive_stats.py", "--store", store, "--work", work,
                                         "--country-map", work / "stats" / "publishers_map.csv"])
    step(5, "the figures of Section 3 and numbers.tex", [PY, "stats/plot_archive.py", "--work", work, "--store", store, "--out", figs])
    step(6, "the query allocation (allocation.tex, allocation_numbers.tex, allocation.json)",
         [PY, "queries/allocate_queries.py", "--stats", work / "stats", *ALLOCATION, "--out-md", out / "query-allocation.md",
          "--out-tex", figs / "allocation.tex", "--out-json", figs / "allocation.json"])
    facts_rc = step(7, "every other own number and the claim checks (facts_numbers.tex)",
                    [PY, "stats/paper_facts.py", "--work", work, "--store", store, "--mirror", mirror,
                     "--mirror-manifest", snap / "mirror_manifest.json", "--runs", runs, "--queries", queries,
                     "--pilot", pilot, "--allocation", figs / "allocation.json", "--out", figs], required=False)
    step(8, "the model rule (model_numbers.tex)", [PY, "model_selection/pareto_simple.py", "--out-report", out / "model_report.md",
                                                   "--out-tex", figs / "model_numbers.tex"])
    # the extraction prompt that the Supplementary Material prints is a copy of the released prompt file
    # (line endings made Unix ones: a checkout on Windows may have turned them into CR LF, which would change the bytes)
    (figs / "draft_needs_prompt.txt").write_bytes((HERE / "queries" / "prompts" / "draft_needs.txt").read_bytes().replace(b"\r\n", b"\n"))
    print("\n=== the extraction prompt: queries/prompts/draft_needs.txt copied to draft_needs_prompt.txt", flush=True)
    # the formula writers' instruction that the Supplementary Material prints is the released English file, from its first
    # numbered heading on
    instruction = (HERE / "queries" / "FORMULA-INSTRUCTION.md").read_bytes().replace(b"\r\n", b"\n")
    (figs / "formula_instruction.txt").write_bytes(instruction[instruction.index(b"## 1. The task"):])
    print("=== the formula writers' instruction: queries/FORMULA-INSTRUCTION.md copied to formula_instruction.txt", flush=True)
    if not args.skip_sources:
        step(9, "the report sources (sources_numbers.tex)", [PY, "queries/prepare_cells.py", "--out", out / "cells",
                                                             *(["--reports", args.reports] if args.reports else []),
                                                             "--allocation", figs / "allocation.json",
                                                             "--numbers-out", figs / "sources_numbers.tex"])
    if args.live:
        step(10, "the public endpoint, today (live_numbers.tex)", [PY, "stats/live_checks.py", "--out", figs, "--report", out])
    backend_checks = []   # (writer, queries, documents, queries whose SPARQL set differs from the index set)
    formulas = Path(args.formulas or snap / "drafting-work" / "queries_with_formulas.json")
    writers = Path(args.writers or snap / "drafting-work" / "writers")
    if formulas.exists():
        for a in ("annotator_a", "annotator_b"):
            step(11, f"the graph run of {a}, remade", [PY, "systems/graph_boolean.py", "--queries", formulas, "--store", store, "--work", work,
                                                     "--author", a, "--node", "cr", "--backend", "index"])
            step(11, f"the graph run of {a} with SPARQL on the store", [PY, "systems/graph_boolean.py", "--queries", formulas, "--store", store,
                                                                       "--work", work, "--author", a, "--node", "cr", "--backend", "sparql",
                                                                       "--tag", f"sparql_{a}_cr"])
            sets_index, sets_sparql = ({r["query_id"]: set(r["doc_ids"])
                                        for r in (json.loads(line) for line in (work / "runs" / f"{tag}_sets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
                                       for tag in (f"graph_{a}_cr", f"sparql_{a}_cr"))
            differ = sorted(q for q in sets_index.keys() | sets_sparql.keys() if sets_index.get(q) != sets_sparql.get(q))
            backend_checks.append((a, len(sets_index), sum(map(len, sets_index.values())), differ))
        step(11, "the writers' numbers (writer_numbers.tex)",
             [PY, "eval/writer_agreement.py", "--work", work, "--queries", formulas, "--authors", "annotator_a,annotator_b",
              "--tex", figs / "writer_numbers.tex", "--writers", writers])
    results_work = Path(args.results_work or snap / "runs-main")
    if (results_work / "eval" / "metrics_ai.json").exists():
        step(12, "the Results section (results_numbers.tex, results_*.tex, results_*.pdf), from the saved result files",
             [PY, "eval/paper_results.py", "--work", results_work, "--queries", formulas, "--out", figs])

    # ---- compare ----
    made = {p.name: p for p in figs.iterdir() if p.is_file()}
    bad = []
    if figures is None:
        print(f"\n=== NO COMPARISON: no folder of the paper's generated files (pass --paper-figures); the {len(made)} remade files are in {figs}")
    else:
        print(f"\n=== THE COMPARISON: remade file vs the file the paper uses ({figures})")
    used = {p.name: p for p in figures.iterdir() if p.is_file()} if figures is not None else {}
    for name in sorted(set(made) | set(used)) if figures is not None else []:
        if name not in used:
            print(f"  NEW        {name} (made, not yet in the paper's folder)")
            continue
        if name not in made:
            skipped = (name == "sources_numbers.tex" and args.skip_sources) or (name in DATED and not args.live)
            print(f"  {'SKIPPED' if skipped else 'MISSING':10} {name}" + ("" if skipped else "  <- the paper uses it but no step made it"))
            if not skipped:
                bad.append(name)
            continue
        if filecmp.cmp(made[name], used[name], shallow=False):
            print(f"  IDENTICAL  {name}")
        else:
            tag = "DATED" if name in DATED else "DIFFERENT"
            print(f"  {tag:10} {name}")
            for line in first_differences(made[name], used[name]):
                print("             " + line)
            if name not in DATED:
                bad.append(name)
    print(f"\n  claim checks (stats/paper_facts.py): {'all PASS' if facts_rc == 0 else 'some FAIL, see ' + str(work / 'stats' / 'paper_facts.md')}")
    print(f"  the claims and their evidence: {work / 'stats' / 'paper_facts.md'}")
    for a, nq, nd, differ in backend_checks:
        print(f"  SPARQL and the entity index give identical sets ({a}: {nq} formulas, {nd} documents): "
              + ("PASS" if not differ else f"FAIL, {len(differ)} queries differ: {differ[:5]}"))
        if differ:
            bad.append(f"SPARQL sets of {a}")
    if args.write:
        for name, p in made.items():
            shutil.copy2(p, figures / name)
        print(f"\n=== WRITE: {len(made)} files copied into {figures}; compiling the paper")
        # pdflatex and bibtex directly, not latexmk: MiKTeX's latexmk needs Perl, which a plain Windows PowerShell lacks
        for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"], ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                    ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "supplementary.tex"],
                    ["pdflatex", "-interaction=nonstopmode", "supplementary.tex"]):
            if not (figures.parent / (cmd[-1] if cmd[-1].endswith(".tex") else cmd[-1] + ".tex")).exists():
                continue
            r = subprocess.run(cmd, cwd=figures.parent, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if r.returncode != 0:
                print(f"    {' '.join(cmd)} failed (exit {r.returncode}); see the .log file in {figures.parent}")
        have_main = (figures.parent / "main.pdf").exists()
        print(f"    compiled: main.pdf {'present' if have_main else 'missing'}"
              + (f", supplementary.pdf {'present' if (figures.parent / 'supplementary.pdf').exists() else 'missing'}"
                 if (figures.parent / "supplementary.tex").exists() else ""))
    if CHECK_TEX.exists():
        print("\n=== THE LATEX: macros defined, numbers typed by hand")
        subprocess.run([PY, str(CHECK_TEX), "--numbers"], cwd=CHECK_TEX.parent)
    total = sum(t for *_, t in log)
    ok = not bad and facts_rc == 0
    print(f"\n=== RESULT: {'EVERYTHING REPRODUCED' if ok else 'NOT REPRODUCED: ' + ', '.join(bad) + ('' if facts_rc == 0 else ' + failed claim checks')}"
          f" ({total / 60:.0f} minutes; remade files in {out})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
