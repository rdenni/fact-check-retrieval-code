"""
run_toy.py -- the end-to-end check: builds the toy data, runs every script of
paper-code on it with the cpu-test profile and fake labels, and fails loudly
if any step breaks. Takes about a minute on a laptop CPU.

Usage: python tests/run_toy.py [--keep]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable


def sh(*cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="keep tests/toy after the run")
    args = ap.parse_args()
    toy = HERE / "toy"
    work = toy / "work"
    Q = toy / "queries_v6.json"
    sh(PY, HERE / "make_toy_data.py", "--out", toy)
    # The snapshot chain without network: dump the toy store to N-Triples, rebuild a store from it with the
    # cleaning rules, and build the publisher map from a two-entry fake IFCN file.
    import pyoxigraph
    st = pyoxigraph.Store(str(toy / "store"))
    with open(toy / "toy_mirror.nt", "wb") as fh:
        st.dump(fh, pyoxigraph.RdfFormat.N_TRIPLES, from_graph=pyoxigraph.DefaultGraph())
    del st
    sh(PY, ROOT / "snapshot" / "build_store.py", "--mirror", toy / "toy_mirror.nt", "--store", toy / "store2")
    (toy / "ifcn_fake.json").write_text(json.dumps([
        {"domain": "snopes.com", "original_label": "Valid IFCN signatory", "original": {"name": "snopes.com", "country": "United States", "language": "English", "website": "https://www.snopes.com", "issued_on": "2025-07-11T00:00:00Z", "expired": False}},
        {"domain": "fullfact.org", "original_label": "Expired IFCN signatory", "original": {"name": "Full Fact", "country": "United Kingdom", "language": "English", "website": "https://fullfact.org", "issued_on": "2023-01-01T00:00:00Z", "expired": True}}]), encoding="utf-8")
    sh(PY, ROOT / "run_all.py", "--queries", Q, "--store", toy / "store", "--work", work, "--profile", "cpu-test", "--appendix", "--node", "claim", "--text-mode", "article")
    # The paper's main runs read the explanation alone: a document without an explanation is not indexed, so no
    # text run may return it, and both indexes must say how many were left out.
    no_article = {d["cr_iri"] for d in map(json.loads, open(work / "corpus.jsonl", encoding="utf-8")) if not (d.get("article_text") or "").strip()}
    bm_params = json.loads((work / "bm25_index" / "params.json").read_text(encoding="utf-8"))
    dense_meta = json.loads((work / "dense_cpu-test" / "meta.json").read_text(encoding="utf-8"))
    returned = {line.split()[2] for tag in ("bm25", "dense", "fused", "pipeline") for line in open(work / "runs" / f"{tag}.trec", encoding="utf-8")}
    if not no_article or bm_params.get("text_mode") != "article" or bm_params.get("n_left_out_no_text") != len(no_article) \
            or dense_meta.get("n_left_out_no_text") != len(no_article) or returned & no_article:
        raise SystemExit(f"explanation mode: the {len(no_article)} documents without an explanation must be left out of both indexes and of every text run")
    for author in ("writer_a", "writer_b", "consensus"):
        sh(PY, ROOT / "systems" / "graph_boolean.py", "--queries", Q, "--store", toy / "store", "--work", work, "--author", author, "--backend", "index", "--node", "claim")
    # The generous version of each writer's formula (systems/ranked_entities.py, judged by the AI only).
    for author in ("writer_a", "writer_b"):
        sh(PY, ROOT / "systems" / "ranked_entities.py", "--queries", Q, "--work", work, "--node", "claim", "--author", author, "--tag", f"relaxed_{author}")
    # The judging plan: the pool for the AI judge and the human check sample (3 toy queries, 12 shared).
    sets_tags = "graph_writer_a_claim,graph_writer_b_claim"
    sh(PY, ROOT / "pool" / "build_pool.py", "--work", work, "--queries", Q, "--ranked", "bm25,pipeline", "--sets", sets_tags,
       "--ai-only", "relaxed_writer_a,relaxed_writer_b", "--judges", "R,M", "--check-queries", "3", "--set-sample", "5",
       "--shared", "12", "--shared-queries", "2")
    check = json.loads((work / "pool" / "check.json").read_text(encoding="utf-8"))
    shared = json.loads((work / "pool" / "shared.json").read_text(encoding="utf-8"))
    assign = json.loads((work / "pool" / "assignments.json").read_text(encoding="utf-8"))
    if len(shared) != 12 or {q for q, _ in shared} - set(check["queries"][:2]):
        raise SystemExit(f"the 12 shared documents should come from the first 2 check queries: {shared}")
    if not all([q, d] in assign["R"] and [q, d] in assign["M"] for q, d in shared):
        raise SystemExit("every shared document must be assigned to both judges")
    if [q for q in dict.fromkeys(x[0] for x in assign["R"])] != [q for q in check["queries"] if any(x[0] == q for x in assign["R"])]:
        raise SystemExit("a judge's documents must follow the order of the check draw")
    pool_rows = [json.loads(line) for line in open(work / "pool" / "pool.jsonl", encoding="utf-8")]
    for q in check["queries"]:
        for s, sample in check["per_query"][q]["set_samples"].items():
            members = next((json.loads(line)["doc_ids"] for line in open(work / "runs" / f"{s}_sets.jsonl", encoding="utf-8")
                            if json.loads(line)["query_id"] == q), [])
            if len(sample) != min(5, len(members)) or set(sample) - set(members):
                raise SystemExit(f"the set sample of {s} for {q} should be 5 members of the set (or all of a smaller set)")
            in_pool = {r["doc_id"] for r in pool_rows if r["query_id"] == q}
            if set(members) - in_pool:
                raise SystemExit(f"every member of the strict set {s} must be in the pool (no cap)")
    # The pool for the tables before a month (pool/extend_pool_before.py): the lists cut to the documents published
    # before 2023-01 or undated, judged as deeply as the whole lists; the main rows and the check sample untouched; a
    # second run changes nothing. metrics.py --before 2023-01 below must then find every scored document judged.
    pool_main = (work / "pool" / "pool.jsonl").read_bytes()
    check_files = {f: (work / "pool" / f).read_bytes() for f in ("check.json", "shared.json", "assignments.json")}
    ext_cmd = (PY, ROOT / "pool" / "extend_pool_before.py", "--work", work, "--before", "2023-01", "--ranked", "bm25,pipeline",
               "--sets", sets_tags, "--ai-only", "relaxed_writer_a,relaxed_writer_b")
    sh(*ext_cmd)
    pool_ext = (work / "pool" / "pool.jsonl").read_bytes()
    ext = json.loads((work / "pool" / "extension_before.json").read_text(encoding="utf-8"))
    if not pool_ext.startswith(pool_main) or ext["added_pairs"] == 0 or len(pool_ext) == len(pool_main) \
            or any((work / "pool" / f).read_bytes() != b for f, b in check_files.items()):
        raise SystemExit("extend_pool_before.py must add pairs after the main pool's rows, unchanged, and leave the check sample alone")
    sh(*ext_cmd)
    if (work / "pool" / "pool.jsonl").read_bytes() != pool_ext:
        raise SystemExit("extend_pool_before.py run twice must give the same pool")
    pool_rows = [json.loads(line) for line in open(work / "pool" / "pool.jsonl", encoding="utf-8")]  # the AI labels all of them
    sh(PY, HERE / "fake_labels.py", "--work", work, "--truth", toy / "truth.json")
    # The optional discussion of the shared documents (not in the paper: the owner's label is final), still tested:
    # decided here from the toy truth.
    sh(PY, ROOT / "judge" / "adjudicate.py", "--work", work, "--annotators", "R,M")
    truth = json.loads((toy / "truth.json").read_text(encoding="utf-8"))
    with open(work / "labels" / "decisions.jsonl", "w", encoding="utf-8") as fh:
        for line in open(work / "labels" / "to_adjudicate.jsonl", encoding="utf-8"):
            r = json.loads(line)
            fh.write(json.dumps({"query_id": r["query_id"], "doc_id": r["doc_id"], "label": "y" if r["doc_id"] in truth[r["query_id"]] else "n"}) + "\n")
    sh(PY, ROOT / "judge" / "adjudicate.py", "--work", work, "--annotators", "R,M", "--decisions", work / "labels" / "decisions.jsonl")
    # The discussion sheet: the same decisions typed into to_discuss.xlsx and read back must give the same labels.
    import openpyxl
    adj_jsonl = (work / "labels" / "labels_adjudicated.jsonl").read_text(encoding="utf-8")
    sh(PY, ROOT / "judge" / "adjudicate.py", "--work", work, "--annotators", "R,M", "--queries", Q, "--discussion-sheet")
    wb = openpyxl.load_workbook(work / "labels" / "to_discuss.xlsx")
    ws = wb.active
    hd = [c.value for c in ws[1]]
    for row in ws.iter_rows(min_row=2):
        q, d = row[hd.index("query_id")].value, row[hd.index("doc_id")].value
        row[hd.index("agreed")].value = "Y" if d in truth[q] else "n"  # a capital letter, as a person may type it
    wb.save(work / "labels" / "decisions.xlsx")
    sh(PY, ROOT / "judge" / "adjudicate.py", "--work", work, "--annotators", "R,M", "--decisions", work / "labels" / "decisions.xlsx")
    if (work / "labels" / "labels_adjudicated.jsonl").read_text(encoding="utf-8") != adj_jsonl:
        raise SystemExit("the decisions read from the discussion sheet must give the same final labels as the JSONL")
    # The AI judge against a fake OpenAI-compatible server (no model needed): the bundle for the judges, the
    # one-command run against the already running fake server (--no-serve), the machine check, the zip.
    srv = subprocess.Popen([PY, str(HERE / "fake_llm_server.py"), "--port", "8123"], cwd=ROOT)
    try:
        time.sleep(1.5)
        sh(PY, ROOT / "judge" / "pack_for_judge.py", "--work", work, "--queries", Q, "--out", work / "judge_bundle", "--zip")
        bundle_pool = (work / "judge_bundle" / "pool" / "pool.jsonl").read_text(encoding="utf-8")
        if '"rating"' in bundle_pool or '"ranks"' in bundle_pool or "provenance" in (work / "judge_bundle" / "queries.json").read_text(encoding="utf-8"):
            raise SystemExit("the bundle must hold only what the prompt uses: no verdict, no system ranks, no provenance")
        (work / "judges_toy.json").write_text(json.dumps([{"name": "fakerun", "model": "fake", "precision": "none"}]), encoding="utf-8")
        nos = ["--no-serve", "--base-url", "http://127.0.0.1:8123/v1", "--concurrency", "4"]
        sh(PY, ROOT / "judge" / "run_judges.py", "run", "--config", work / "judges_toy.json", "--bundle", work / "judge_bundle", *nos)
        # a bundle packed with another prompt must be refused: a copy whose MANIFEST names another fingerprint
        odd = work / "judge_bundle_other_prompt"
        shutil.copytree(work / "judge_bundle", odd, ignore=shutil.ignore_patterns("judge"))
        m = json.loads((odd / "MANIFEST.json").read_text(encoding="utf-8"))
        m["prompt_sha256"] = "0" * 64
        (odd / "MANIFEST.json").write_text(json.dumps(m), encoding="utf-8")
        r = subprocess.run([PY, str(ROOT / "judge" / "run_judges.py"), "run", "--config", str(work / "judges_toy.json"), "--bundle", str(odd), *nos],
                           capture_output=True, text=True, cwd=ROOT)
        if r.returncode == 0 or "packed with the prompt sha256" not in (r.stderr + r.stdout):
            raise SystemExit("run_judges.py must refuse a bundle packed with another prompt:\n" + r.stdout + r.stderr)
        shutil.rmtree(odd)
        # An API judge, as in judge/judges_api.json: a base_url in the model list (no server is started), the key read
        # from an environment variable and sent, the routing fields passed through, and every label recording who
        # answered and what it cost. Then an account without credit (HTTP 402) must stop the run at once.
        api = {"name": "fakeapi", "model": "fake-api", "precision": "none", "base_url": "http://127.0.0.1:8123/v1",
               "api_key_env": "TOY_JUDGE_KEY", "concurrency": 4,
               "extra": {"provider": {"order": ["fakeprovider"], "allow_fallbacks": False}}}
        (work / "judges_toy_api.json").write_text(json.dumps([api]), encoding="utf-8")
        os.environ["TOY_JUDGE_KEY"] = "toy-key"
        sh(PY, ROOT / "judge" / "run_judges.py", "run", "--config", work / "judges_toy_api.json", "--bundle", work / "judge_bundle",
           "--max-items", "3")
        api_rows = [json.loads(x) for x in (work / "judge_bundle" / "judge" / "labels_llm_fakeapi.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(api_rows) != 3 or {x["provider"] for x in api_rows} != {"FakeProvider"} or any(x["cost"] != 0.0001 for x in api_rows):
            raise SystemExit("the API judge must label 3 pairs (--max-items) and record the provider and the cost of each")
        sh(PY, ROOT / "judge" / "run_judges.py", "run", "--config", work / "judges_toy_api.json", "--bundle", work / "judge_bundle")
        if json.loads((work / "judge_bundle" / "judge" / "run_fakeapi.json").read_text(encoding="utf-8"))["labels"] != len(pool_rows):
            raise SystemExit("the API judge's second run must resume and label every pair of the pool")
        (work / "judges_toy_nocredit.json").write_text(json.dumps([{**api, "name": "fakenocredit", "model": "fake-nocredit"}]), encoding="utf-8")
        r = subprocess.run([PY, str(ROOT / "judge" / "run_judges.py"), "run", "--config", str(work / "judges_toy_nocredit.json"),
                            "--bundle", str(work / "judge_bundle")], capture_output=True, text=True, cwd=ROOT)
        if r.returncode == 0 or "HTTP 402" not in (r.stdout + r.stderr):
            raise SystemExit("an account without credit (HTTP 402) must stop the judge with a clear message:\n" + r.stdout + r.stderr)
        if (work / "judge_bundle" / "judge" / "labels_llm_fakenocredit.jsonl").exists():
            raise SystemExit("a refused run must write no label")
        for f in ("run_fakenocredit.json",):
            (work / "judge_bundle" / "judge" / f).unlink(missing_ok=True)
    finally:
        srv.terminate()
    run_info = json.loads((work / "judge_bundle" / "judge" / "run_fakerun.json").read_text(encoding="utf-8"))
    if run_info["labels"] != run_info["pairs_in_pool"] or run_info["labels"] != len(pool_rows):
        raise SystemExit(f"the AI judge should label every pair of the pool: {run_info['labels']} of {len(pool_rows)}")
    sh(PY, ROOT / "judge" / "run_judges.py", "check", "--out", work / "judge_check")
    sh(PY, ROOT / "judge" / "run_judges.py", "pack", "--bundle", work / "judge_bundle", "--out", work / "judge_results.zip",
       "--add", work / "judge_check" / "env.json")
    # The answer parser of the AI judge: one word, YES or NO.
    sys.path.insert(0, str(ROOT))
    from judge.llm_client import parse_yes_no
    if [parse_yes_no(x) for x in ("YES", "No.", "Answer: YES", '{"label": "Not Relevant"}', "maybe")] != ["y", "n", "y", "n", None]:
        raise SystemExit("parse_yes_no reads the judge's answer wrongly")
    sh(PY, ROOT / "judge" / "validate_judge.py", "--work", work, "--annotators", "R,M")
    validation = json.loads((work / "judge" / "validation.json").read_text(encoding="utf-8"))
    if "majority" not in validation["candidates"] or not validation["main_judge"] or validation["human_human"]["bootstrap_95"][0] is None:
        raise SystemExit(f"validation should build the majority candidate, the bootstrap interval and pick a main judge: {validation.get('main_judge')}")
    # The drafting chain (queries/draft_needs.py, dedup_needs.py, review_needs.py, draw_queries.py) against the same
    # fake server: a toy cell with two documents, candidates, a lexical merge, a review sheet filled with K by this
    # test, the blind sample and its agreement, the seeded draw against a toy allocation, the rewrite round trip.
    cells = toy / "cells"
    cells.mkdir(exist_ok=True)
    (cells / "us-2020.md").write_text(
        "# Cell United States 2020: 2 queries, 2 documents (tiers 1)\n\n\n---\n\n## DOCUMENT toy-report-a\n"
        "tier: 1 | publisher type: research-institute | years: 2020 | url: https://example.org/a\nfile: us/toy-report-a.html | sha256: 0000\n\n"
        "The report records the claim that mail ballots were counted twice in Pine County during the 2020 vote. "
        "It also records the claim that the Pine County clerk shredded ballots after the count ended. "
        "Monitoring found a troll farm amplifying hashtags in October.\n\n\n---\n\n## DOCUMENT toy-report-b\n"
        "tier: 1 | publisher type: official | years: 2020 | url: https://example.org/b\nfile: us/toy-report-b.html | sha256: 0001\n\n"
        "A register entry repeats the claim that mail ballots were counted twice in Pine County during the 2020 vote. "
        "Another entry records the claim that voting machines in Oak City switched votes overnight.\n", encoding="utf-8")
    srv = subprocess.Popen([PY, str(HERE / "fake_llm_server.py"), "--port", "8124"], cwd=ROOT)
    try:
        time.sleep(1.5)
        sh(PY, ROOT / "queries" / "draft_needs.py", "--cells", cells, "--work", work, "--base-url", "http://127.0.0.1:8124/v1",
           "--model", "fake", "--precision", "none")
        # the stability measurement: the same requests asked again (the fake server answers the same, so 100 percent);
        # its script is not part of the shared code, so the step runs only where it exists
        if (ROOT / "queries" / "stability_needs.py").exists():
            sh(PY, ROOT / "queries" / "stability_needs.py", "--cells", cells, "--work", work, "--base-url", "http://127.0.0.1:8124/v1",
               "--model", "fake", "--precision", "none", "--n", "2", "--method", "lexical", "--threshold", "0.6")
    finally:
        srv.terminate()
    sh(PY, ROOT / "queries" / "dedup_needs.py", "--work", work, "--method", "lexical", "--threshold", "0.6")
    (toy / "allocation_toy.json").write_text('{"cells": {"United States": {"2020": 1}}}', encoding="utf-8")
    # the top-up targets: the closed years of a country (no row left) miss 3 + 1 queries, so its two open
    # years must find 4 more K, shared in proportion to their rows (30 and 10: 3 and 1)
    sys.path.insert(0, str(ROOT / "queries"))
    import review_needs
    t = review_needs.extended_targets({"us-2016": {"rows": 2, "reviewed_from_top": 2, "K_from_top": 1},
                                       "us-2020": {"rows": 30, "reviewed_from_top": 5, "K_from_top": 3},
                                       "us-2021": {"rows": 10, "reviewed_from_top": 4, "K_from_top": 3}},
                                      {"us-2016": 4, "us-2017": 1, "us-2020": 2, "us-2021": 1}, 2)
    if t != {"us-2016": 2, "us-2020": 7, "us-2021": 4}:
        raise SystemExit(f"extended_targets wrong: {t}")
    # rule D3 needs a kept twin the draw uses: a twin in another country is fine; a twin that is itself a
    # drop is an error; a D3 row below a gap (not used by the draw) is not checked
    e = review_needs.d3_twin_errors([
        {"cluster_id": "us-2020:c0001", "decision": "K", "rule": "", "note": "", "eligible": True},
        {"cluster_id": "south-africa-2020:c0002", "decision": "D", "rule": "D3", "note": "same as us-2020:c0001", "eligible": True},
        {"cluster_id": "us-2020:c0003", "decision": "D", "rule": "D4", "note": "", "eligible": True},
        {"cluster_id": "us-2020:c0004", "decision": "D", "rule": "D3", "note": "us-2020:c0003", "eligible": True},
        {"cluster_id": "us-2020:c0005", "decision": "D", "rule": "D3", "note": "", "eligible": False}], True)
    if len(e) != 1 or not e[0].startswith("us-2020:c0004") or "us-2020:c0003: D" not in e[0]:
        raise SystemExit(f"d3_twin_errors wrong: {e}")
    # Cochran's procedure: C would get 2 of 10 but can supply 1; the other 9 are split again (A 6, B 3)
    import allocate_queries
    a = allocate_queries.capped({"A": 50.0, "B": 30.0, "C": 20.0}, 10, 0, {"C": 1})
    if a != {"A": 6, "B": 3, "C": 1}:
        raise SystemExit(f"capped allocation wrong: {a}")
    # The review in quota mode: a first plain sheet with one decision, carried into the random-order sheet;
    # the reviewer fills the cell from its top only until the target (1 query + margin 1), the draw refuses an unfinished
    # cell, then accepts it once the target is reached; the rows below the target stay empty.
    import openpyxl
    quota_args = ["--allocation", toy / "allocation_toy.json", "--margin", "1"]
    sh(PY, ROOT / "queries" / "review_needs.py", "sheet", "--work", work, "--out", work / "drafting" / "review_quota.xlsx", *quota_args)
    ws = openpyxl.load_workbook(work / "drafting" / "review_quota.xlsx").active
    hq = [c.value for c in ws[1]]
    last_in_order = ws.cell(row=ws.max_row, column=hq.index("cluster_id") + 1).value  # carried below the stop: logged, not drawn
    sh(PY, ROOT / "queries" / "review_needs.py", "sheet", "--work", work, "--out", work / "drafting" / "review.xlsx")
    wb = openpyxl.load_workbook(work / "drafting" / "review.xlsx")
    ws = wb.active
    header = [c.value for c in ws[1]]
    for r in range(2, ws.max_row + 1):
        if ws.cell(row=r, column=header.index("cluster_id") + 1).value == last_in_order:
            ws.cell(row=r, column=header.index("decision") + 1).value = "K"
    wb.save(work / "drafting" / "review.xlsx")
    sh(PY, ROOT / "queries" / "review_needs.py", "sheet", "--work", work, "--out", work / "drafting" / "review_quota.xlsx", *quota_args,
       "--carry", work / "drafting" / "review.xlsx")

    def fill(n_k):  # write K from the top of the random order until the cell has n_k K (the carried decision counts)
        wb = openpyxl.load_workbook(work / "drafting" / "review_quota.xlsx")
        ws = wb.active
        hq = [c.value for c in ws[1]]
        k = 0
        for r in range(2, ws.max_row + 1):
            if k >= n_k:
                break
            c = ws.cell(row=r, column=hq.index("decision") + 1)
            c.value = c.value or "K"
            k += c.value == "K"
        wb.save(work / "drafting" / "review_quota.xlsx")
    fill(1)
    sh(PY, ROOT / "queries" / "review_needs.py", "import", "--work", work, "--sheet", work / "drafting" / "review_quota.xlsx", *quota_args)
    r = subprocess.run([PY, str(ROOT / "queries" / "draw_queries.py"), "draw", "--work", str(work), "--allocation", str(toy / "allocation_toy.json"),
                        "--seed", "1", "--out", str(work / "queries_drawn.json")], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 1 or "not finished" not in r.stdout:
        raise SystemExit("the draw should refuse an unfinished quota-mode review:\n" + r.stdout + r.stderr)
    fill(2)
    sh(PY, ROOT / "queries" / "review_needs.py", "import", "--work", work, "--sheet", work / "drafting" / "review_quota.xlsx", *quota_args)
    sh(PY, ROOT / "queries" / "review_needs.py", "second", "--work", work, "--out", work / "drafting" / "second.xlsx", "--fraction", "1.0")
    wb = openpyxl.load_workbook(work / "drafting" / "second.xlsx")
    ws = wb.active
    h2 = [c.value for c in ws[1]]
    for r in range(2, ws.max_row + 1):
        ws.cell(row=r, column=h2.index("decision") + 1).value = "K"
    wb.save(work / "drafting" / "second.xlsx")
    sh(PY, ROOT / "queries" / "review_needs.py", "agreement", "--work", work, "--sheet", work / "drafting" / "second.xlsx")
    sh(PY, ROOT / "queries" / "draw_queries.py", "draw", "--work", work, "--allocation", toy / "allocation_toy.json", "--seed", "1",
       "--pilot", "1", "--pool-short", "--out", work / "queries_drawn.json")
    main_ids = {q["provenance"]["cluster_id"] for q in json.load(open(work / "queries_drawn.json", encoding="utf-8"))}
    pilot_ids = {q["provenance"]["cluster_id"] for q in json.load(open(work / "queries_pilot.json", encoding="utf-8"))}
    if len(pilot_ids) != 1 or main_ids & pilot_ids:
        raise SystemExit(f"pilot draw wrong: main {main_ids}, pilot {pilot_ids}")
    # the model's proposals, from the fake server, then the sheet with them copied into free_text
    srv = subprocess.Popen([PY, str(HERE / "fake_llm_server.py"), "--port", "8124"], cwd=ROOT)
    try:
        time.sleep(1.5)
        sh(PY, ROOT / "queries" / "propose_rewrites.py", "--queries", work / "queries_drawn.json", "--work", work,
           "--base-url", "http://127.0.0.1:8124/v1", "--model", "fake", "--precision", "none", "--no-think")
    finally:
        srv.terminate()
    sh(PY, ROOT / "queries" / "draw_queries.py", "rewrite-sheet", "--queries", work / "queries_drawn.json", "--work", work,
       "--proposals", work / "drafting" / "rewrite_proposals_queries_drawn.jsonl", "--out", work / "drafting" / "rewrite.xlsx")
    wb = openpyxl.load_workbook(work / "drafting" / "rewrite.xlsx")
    ws = wb.active
    hdr = [c.value for c in ws[1]]
    if not all(ws.cell(row=r, column=hdr.index("free_text") + 1).value for r in range(2, ws.max_row + 1)):
        raise SystemExit("the rewrite sheet should carry the proposals in free_text")
    if "proposal" in hdr:
        raise SystemExit("the rewrite sheet should not show the local model's proposal column")
    # the rewrite checks on made-up rows: the country's possessive and a month pass; an added name and a verdict word do not
    import draw_queries
    q = {"stratum": {"country": "India", "year": 2020}, "draft_text": "", "provenance": {
        "claim": "Hot weather kills the virus", "passage": "In 2020 it was falsely claimed that hot weather kills the virus"}}
    ok_text = "India's hot weather kills the virus, as claimed in May 2020"
    if draw_queries.rewrite_problems(ok_text, q):
        raise SystemExit(f"rewrite check too strict: {draw_queries.rewrite_problems(ok_text, q)}")
    if draw_queries.rewrite_problems("Catholic weather kills the virus", q, "the Anglican-Catholic rivalry"):  # a hyphenated word counts by its pieces
        raise SystemExit("a name inside a hyphenated word of the context should pass")
    if draw_queries.rewrite_problems("An hour of hot weather kills the virus", q):  # the passage has no "an"
        raise SystemExit("a capitalised first word such as 'An' is not a name: " + str(draw_queries.rewrite_problems("An hour of hot weather kills the virus", q)))
    if len(draw_queries.rewrite_problems("Hot weather kills the virus, a false claim by Modi", q)) != 2:
        raise SystemExit("rewrite check should flag the added name and the verdict word")
    # W5 reworded: a background's names must be in the passage, the claim or the context; no verdict word;
    # it may say who said what
    if not draw_queries.background_problems("Roy Moore, a Republican candidate", q, ""):
        raise SystemExit("a background name in no source should be flagged")
    if draw_queries.background_problems("Roy Moore, who said that he was a Republican candidate", q, "Republican Roy Moore"):
        raise SystemExit("a background name from the context should pass, and a background may say who said what")
    if not draw_queries.background_problems("a false claim about the virus", q, ""):
        raise SystemExit("a verdict word in a background should be flagged")
    # no quotation marks in a query; an apostrophe is not one; a background is not checked for them
    for quoted in ("Hot weather ‘kills’ the virus", "Hot weather “kills” the virus", "Hot weather 'kills' the virus", "'Hot' weather kills the virus"):
        if not any(w.startswith("quotation marks") for w in draw_queries.rewrite_problems(quoted, q)):
            raise SystemExit(f"a quotation mark in a query should be flagged: {quoted}")
    if draw_queries.rewrite_problems("India’s hot weather kills the farmers’ virus, and India's too", q):
        raise SystemExit("an apostrophe is not a quotation mark: " + str(draw_queries.rewrite_problems("India’s hot weather kills the farmers’ virus, and India's too", q)))
    if draw_queries.background_problems("the virus ‘kills’", q, ""):
        raise SystemExit("a background is not checked for quotation marks")
    # no hyphens or dashes in a query; a background may have them
    for dashed in ("Hot weather kills the COVID-19 virus", "Hot weather kills the pro-virus camp", "Hot weather kills the virus on 9–10 May", "Hot weather - it kills the virus"):
        if not any(w.startswith("hyphens") for w in draw_queries.rewrite_problems(dashed, q)):
            raise SystemExit(f"a hyphen or dash in a query should be flagged: {dashed}")
    if draw_queries.rewrite_problems("Hot weather kills the 19 viruses of 2020", q) or draw_queries.background_problems("a 21st-century virus", q, ""):
        raise SystemExit("a query without hyphens, and a background with one, should pass")
    # rules W4 and D3: two queries stating the same claim after the pick are reported, with the one to replace
    same = draw_queries.same_statements([("a-1", "Muslims are violent", True), ("b-1", "Hot weather kills the virus.", False),
                                         ("c-1", "muslims are VIOLENT", True), ("d-1", "Hot weather kills the virus", True), ("e-1", "Cold kills", False)])
    if len(same) != 2 or "replace c-1" not in same[0] or "replace d-1" not in same[1]:
        raise SystemExit(f"same_statements should report the two pairs and replace the later or the split one: {same}")
    # the context copied from the report allows its names and years; a year in no source is flagged
    if draw_queries.rewrite_problems("Hot weather kills the virus (Kerala, March 2021)", q, "Kerala, 1 March 2021"):
        raise SystemExit("rewrite check should accept a place and a year from the context")
    if len(draw_queries.rewrite_problems("Hot weather kills the virus (2019)", q, "", "2020-2021")) != 1:
        raise SystemExit("rewrite check should flag a year in neither the passage, the context nor the period")
    rep = work / "toy_report.txt"
    rep.write_text("E l e c t i o n  W e e k  2 0 1 6\nNovember 7,\n2016 an account tweeted", encoding="utf-8")
    if draw_queries.context_not_in_report("Election Week 2016 … November 7, 2016", rep.with_suffix(".pdf")):
        raise SystemExit("a context copied from the report should be found in it")
    if not draw_queries.context_not_in_report("November 8, 2016", rep.with_suffix(".pdf")):
        raise SystemExit("a context not in the report should be flagged")
    # the proposal is the statement alone; the place and the date are a hint: a place the passage names is
    # kept, one it does not name is left out and noted; a time-bound claim with no date of its own gets the period's year
    import propose_rewrites
    got = propose_rewrites.compose({"statement": "Farmers threw manure at an embassy.", "place": "Paris", "when": "", "time_bound": True},
                                   "it was claimed that farmers threw manure at an embassy in Paris", "2024")
    if got != ("Farmers threw manure at an embassy.", [], "Paris, 2024"):
        raise SystemExit(f"compose wrong: {got}")
    got = propose_rewrites.compose({"statement": "Asylum seekers can vote.", "place": "Ireland", "when": "", "time_bound": False},
                                   "in Spain it was claimed that asylum seekers can vote", "2023")
    if got[0] != "Asylum seekers can vote." or not got[1] or got[2]:
        raise SystemExit(f"compose should leave out a place the passage does not name: {got}")
    for r in range(2, ws.max_row + 1):
        ws.cell(row=r, column=hdr.index("free_text") + 1).value = "Were " + str(ws.cell(row=r, column=hdr.index("claim") + 1).value)
    ws.cell(row=2, column=hdr.index("context") + 1).value = "a context copied from the report"
    # a background for the judges, released with the query, never searched
    ws.cell(row=2, column=hdr.index("background") + 1).value = "a background copied from the report"
    # rule W4: a claim that joins two claims; the import picks one part with the draw's seed
    ws.cell(row=2, column=hdr.index("parts") + 1).value = "mail ballots were counted twice || the clerk shredded ballots"
    wb.save(work / "drafting" / "rewrite.xlsx")
    sh(PY, ROOT / "queries" / "draw_queries.py", "import-rewrite", "--queries", work / "queries_drawn.json", "--sheet", work / "drafting" / "rewrite.xlsx",
       "--proposals", work / "drafting" / "rewrite_proposals_queries_drawn.jsonl", "--reports", work, "--out", work / "queries_v7.json")
    v7 = json.loads((work / "queries_v7.json").read_text(encoding="utf-8"))
    if not all(x["provenance"].get("rewrite_proposal") for x in v7):
        raise SystemExit("import-rewrite --proposals should record the local model's proposal, which the sheet no longer shows")
    if not any(x["provenance"].get("rewrite_context") for x in v7):
        raise SystemExit("import-rewrite should keep the context")
    with_bg = [x for x in v7 if x.get("background") == "a background copied from the report"]
    if len(with_bg) != 1:
        raise SystemExit("import-rewrite should store the background as a query field")
    sys.path.insert(0, str(ROOT))
    from judge import prompts as judge_prompts
    item = {"claim_text": "a claim", "date_published": "2024-01-01", "author_name": "Toy Check", "article_start": ""}
    with_line, _ = judge_prompts.judge_prompt(with_bg[0], item)
    without, _ = judge_prompts.judge_prompt({"free_text": "x"}, item)
    if "\nBackground: a background copied from the report\n" not in with_line or "\nBackground: " in without \
            or "\nStart of the explanation: (none)\n" not in without or "country: ?" not in without:
        raise SystemExit("the AI judge's prompt should carry the background only when there is one, and always the "
                         "explanation line and the fact-checker's country")
    if not any(x["provenance"].get("rewrite_part_kept") in (1, 2) and len(x["provenance"].get("rewrite_parts", [])) == 2 for x in v7):
        raise SystemExit("import-rewrite should record the two parts and the part kept (W4)")
    if draw_queries.pick_part(["a", "b", "c"], 20261010, "us-2016-03") != draw_queries.pick_part(["a", "b", "c"], 20261010, "us-2016-03"):
        raise SystemExit("pick_part must depend only on the seed and the query id")
    if draw_queries.part_scores("mail ballots were counted twice", ["mail ballots counted twice", "the clerk shredded ballots"])[0] != 1.0:
        raise SystemExit("part_scores wrong")
    # W4: the parts are fixed at the first import; reordering them afterwards is flagged (on a copy, so nothing else changes)
    if not (work / "queries_v7.w4.json").exists():
        raise SystemExit("import-rewrite should remember the W4 parts in queries_v7.w4.json")
    shutil.copy(work / "queries_v7.w4.json", work / "w4test.w4.json")
    wb2 = openpyxl.load_workbook(work / "drafting" / "rewrite.xlsx")
    wb2.active.cell(row=2, column=hdr.index("parts") + 1).value = "the clerk shredded ballots || mail ballots were counted twice"
    wb2.save(work / "drafting" / "rewrite_w4test.xlsx")
    sh(PY, ROOT / "queries" / "draw_queries.py", "import-rewrite", "--queries", work / "queries_drawn.json", "--sheet",
       work / "drafting" / "rewrite_w4test.xlsx", "--reports", work, "--out", work / "w4test.json")
    if "the parts changed after an import" not in (work / "w4test.rewrite.md").read_text(encoding="utf-8"):
        raise SystemExit("import-rewrite should flag parts reordered after the pick was shown")
    # --keep: rebuilding the sheet keeps the reviewer's parts, free_text and context for an unchanged claim
    sh(PY, ROOT / "queries" / "draw_queries.py", "rewrite-sheet", "--queries", work / "queries_drawn.json", "--work", work,
       "--proposals", work / "drafting" / "rewrite_proposals_queries_drawn.jsonl", "--keep", work / "drafting" / "rewrite.xlsx",
       "--out", work / "drafting" / "rewrite.xlsx")
    ws = openpyxl.load_workbook(work / "drafting" / "rewrite.xlsx").active
    if not (str(ws.cell(row=2, column=hdr.index("free_text") + 1).value).startswith("Were ")
            and ws.cell(row=2, column=hdr.index("parts") + 1).value and ws.cell(row=2, column=hdr.index("context") + 1).value
            and ws.cell(row=2, column=hdr.index("background") + 1).value):
        raise SystemExit("rewrite-sheet --keep should carry over free_text, parts and context")
    # The closed model's proposals: a separate file fills a row nobody has written; a rebuild never overwrites the
    # reviewer's correction; the import keeps the closed model's proposal beside the final text and counts the changes
    qd = json.loads((work / "queries_drawn.json").read_text(encoding="utf-8"))
    cl = work / "drafting" / "rewrite_closed_queries_drawn.jsonl"
    cl.write_text(json.dumps({"query_id": qd[-1]["query_id"], "cluster_id": qd[-1]["provenance"]["cluster_id"], "free_text": "the closed model's sentence",
                              "context": "", "background": "", "parts": "", "note": "why", "model": "closed-model-x"}) + "\n", encoding="utf-8")
    sh(PY, ROOT / "queries" / "draw_queries.py", "rewrite-sheet", "--queries", work / "queries_drawn.json", "--work", work,
       "--proposals", work / "drafting" / "rewrite_proposals_queries_drawn.jsonl", "--closed-proposals", cl, "--out", work / "drafting" / "rewrite_closed.xlsx")
    wbc = openpyxl.load_workbook(work / "drafting" / "rewrite_closed.xlsx")
    wsc, hc = wbc.active, [c.value for c in wbc.active[1]]
    last = wsc.max_row
    if wsc.cell(row=last, column=hc.index("free_text") + 1).value != "the closed model's sentence" or "closed_note" in hc \
            or not str(wsc.cell(row=last, column=hc.index("reworded") + 1).value or "").startswith("X"):
        raise SystemExit("rewrite-sheet --closed-proposals should put the closed model's proposal into a row nobody has written, mark it reworded (X), and show no closed_note")
    wsc.cell(row=last, column=hc.index("free_text") + 1).value = "The reviewer's sentence about Zorblax"
    wsc.cell(row=last, column=hc.index("background") + 1).value = "Zorblax, a made-up name"
    wbc.save(work / "drafting" / "rewrite_closed.xlsx")
    sh(PY, ROOT / "queries" / "draw_queries.py", "rewrite-sheet", "--queries", work / "queries_drawn.json", "--work", work,
       "--proposals", work / "drafting" / "rewrite_proposals_queries_drawn.jsonl", "--closed-proposals", cl, "--keep", work / "drafting" / "rewrite_closed.xlsx",
       "--out", work / "drafting" / "rewrite_closed.xlsx")
    wsc = openpyxl.load_workbook(work / "drafting" / "rewrite_closed.xlsx").active
    if wsc.cell(row=last, column=hc.index("free_text") + 1).value != "The reviewer's sentence about Zorblax":
        raise SystemExit("a rebuild with --keep and --closed-proposals must not overwrite the reviewer's correction")
    # an edited claim cell: the row's work is not carried over, and the rebuild says so
    wbe = openpyxl.load_workbook(work / "drafting" / "rewrite_closed.xlsx")
    wbe.active.cell(row=2, column=hc.index("claim") + 1).value = "an edited claim"
    wbe.save(work / "drafting" / "rewrite_edited_claim.xlsx")
    out = subprocess.run([str(PY), str(ROOT / "queries" / "draw_queries.py"), "rewrite-sheet", "--queries", str(work / "queries_drawn.json"),
                          "--work", str(work), "--keep", str(work / "drafting" / "rewrite_edited_claim.xlsx"),
                          "--out", str(work / "drafting" / "rewrite_edited_claim_rebuilt.xlsx")], check=True, cwd=ROOT, capture_output=True, text=True).stdout
    if "NOT carried over" not in out or str(wsc.cell(row=2, column=1).value) not in out.split("NOT carried over")[1]:
        raise SystemExit("a rebuild must name a row whose claim cell was edited: " + out)
    # a context_only entry (the closed model wrote only the context of a row the reviewer wrote) is never put into the
    # sheet, and the import records it without counting it as a proposed sentence
    co = work / "drafting" / "rewrite_closed_context_only.jsonl"
    co.write_text(json.dumps({"query_id": qd[-1]["query_id"], "cluster_id": qd[-1]["provenance"]["cluster_id"], "free_text": "", "context": "a context",
                              "background": "", "parts": "", "note": "Context filled by the closed model", "model": "closed-model-x", "context_only": True}) + "\n", encoding="utf-8")
    sh(PY, ROOT / "queries" / "draw_queries.py", "rewrite-sheet", "--queries", work / "queries_drawn.json", "--work", work,
       "--proposals", work / "drafting" / "rewrite_proposals_queries_drawn.jsonl", "--closed-proposals", co, "--out", work / "drafting" / "rewrite_co.xlsx")
    if openpyxl.load_workbook(work / "drafting" / "rewrite_co.xlsx").active.cell(row=last, column=hc.index("context") + 1).value:
        raise SystemExit("rewrite-sheet must not put a context_only entry into the sheet")
    sh(PY, ROOT / "queries" / "draw_queries.py", "import-rewrite", "--queries", work / "queries_drawn.json", "--sheet", work / "drafting" / "rewrite_closed.xlsx",
       "--closed-proposals", co, "--out", work / "closed_co.json")
    cc = {x["query_id"]: x for x in json.loads((work / "closed_co.json").read_text(encoding="utf-8"))}[qd[-1]["query_id"]]
    if not cc["provenance"].get("rewrite_closed", {}).get("context_only") or "0 of 0 of the closed model's sentences" not in (work / "closed_co.rewrite.md").read_text(encoding="utf-8"):
        raise SystemExit("import-rewrite should record a context_only entry without counting it as a proposed sentence")
    sh(PY, ROOT / "queries" / "draw_queries.py", "import-rewrite", "--queries", work / "queries_drawn.json", "--sheet", work / "drafting" / "rewrite_closed.xlsx",
       "--closed-proposals", cl, "--out", work / "closed_test.json")
    ct = {x["query_id"]: x for x in json.loads((work / "closed_test.json").read_text(encoding="utf-8"))}[qd[-1]["query_id"]]
    crep = (work / "closed_test.rewrite.md").read_text(encoding="utf-8")
    if ct["free_text"] != "The reviewer's sentence about Zorblax" or ct["provenance"].get("rewrite_closed", {}).get("free_text") != "the closed model's sentence" \
            or "1 of 1 of the closed model's sentences changed" not in crep:
        raise SystemExit("import-rewrite --closed-proposals should keep the closed model's proposal beside the final text and count the change")
    # W5 reworded: a name given only by the background is no longer a receipt for the text, and the
    # background's own names must be in the context
    if "names not in the passage or the context: Zorblax" not in crep or "background: names not in the passage or the context: Zorblax" not in crep:
        raise SystemExit("import-rewrite should flag a name that only the background gives, in the text and in the background")
    # rule Replacement, on a copy: the pool is the unused kept rows of the cell (the country for a top-up), the
    # replaced row must be D in the log, the other queries do not change, the old file is kept, the replacement is logged
    rw = toy / "replace_work"
    (rw / "drafting").mkdir(parents=True, exist_ok=True)
    log = [json.loads(l) for l in open(work / "drafting" / "reviewed.jsonl", encoding="utf-8")]
    main_q = json.loads((work / "queries_drawn.json").read_text(encoding="utf-8"))
    target = main_q[0]
    for r in log:
        if r["cluster_id"] == target["provenance"]["cluster_id"]:
            r["decision"], r["rule"] = "D", "D4"
    spare = [dict(log[0], cluster_id=f"{target['provenance']['cluster_id'].split(':')[0]}:c90{i}", decision="K", eligible=True) for i in (1, 2)]
    with open(rw / "drafting" / "reviewed.jsonl", "w", encoding="utf-8") as fh:
        for r in log + spare:
            fh.write(json.dumps(r) + "\n")
    (rw / "queries_drawn.json").write_text(json.dumps(main_q), encoding="utf-8")
    sh(PY, ROOT / "queries" / "draw_queries.py", "replace", "--queries", rw / "queries_drawn.json", "--other", work / "queries_pilot.json",
       "--work", rw, "--query-id", target["query_id"], "--out", rw / "queries_drawn.json")
    after = json.loads((rw / "queries_drawn.json").read_text(encoding="utf-8"))
    new = next(x for x in after if x["query_id"] == target["query_id"])
    kept_free = {r["cluster_id"] for r in log + spare if r["decision"] == "K" and r.get("eligible", True)} - pilot_ids - main_ids
    if new["provenance"]["cluster_id"] not in kept_free or new["provenance"]["replaces"] != [target["provenance"]["cluster_id"]]:
        raise SystemExit(f"replace drew from the wrong pool: {new['provenance']['cluster_id']}")
    if [x["query_id"] for x in after] != [x["query_id"] for x in main_q] or not (rw / "drafting" / "replacements.jsonl").exists() \
            or not list(rw.glob("queries_drawn.before_replace_*.json")):
        raise SystemExit("replace should keep the other queries, the old file and a log line")
    top = dict(target, provenance=dict(target["provenance"], drawn_from="top-up of us-2017 from the other years of United States"))
    pool, scope = draw_queries.replacement_pool(top, log + spare, set())
    if "country" not in scope:
        raise SystemExit("a top-up query is replaced from its country")
    # no unused kept row in the country either (every row D on a second copy): the query is removed and logged
    rw2 = toy / "replace_work2"
    (rw2 / "drafting").mkdir(parents=True, exist_ok=True)
    with open(rw2 / "drafting" / "reviewed.jsonl", "w", encoding="utf-8") as fh:
        for r in log:
            fh.write(json.dumps(dict(r, decision="D", rule="D4")) + "\n")
    (rw2 / "queries_drawn.json").write_text(json.dumps(main_q), encoding="utf-8")
    sh(PY, ROOT / "queries" / "draw_queries.py", "replace", "--queries", rw2 / "queries_drawn.json", "--other", work / "queries_pilot.json",
       "--work", rw2, "--query-id", target["query_id"], "--out", rw2 / "queries_drawn.json")
    gone = json.loads((rw2 / "queries_drawn.json").read_text(encoding="utf-8"))
    removal = json.loads((rw2 / "drafting" / "replacements.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    if any(x["query_id"] == target["query_id"] for x in gone) or len(gone) != len(main_q) - 1 or removal["new"] is not None:
        raise SystemExit("replace should remove the query and log it when the country has no unused kept row")
    again = dict(top, provenance=dict(top["provenance"], drawn_from=top["provenance"]["drawn_from"] + "; replacement of x (D4)"))
    if "country" not in draw_queries.replacement_pool(again, log + spare, set())[1]:
        raise SystemExit("a top-up query replaced once is replaced from its country again")
    if not new["provenance"]["drawn_from"].startswith("replacement of") and target["provenance"].get("drawn_from") is None:
        raise SystemExit("replace should write drawn_from")
    # a sheet row written for another claim (a replaced query) is refused by the import, never imported silently
    r = subprocess.run([PY, str(ROOT / "queries" / "draw_queries.py"), "import-rewrite", "--queries", str(rw / "queries_drawn.json"),
                        "--sheet", str(work / "drafting" / "rewrite.xlsx"), "--out", str(rw / "queries_v7.json")], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 1 or "another claim" not in (rw / "queries_v7.rewrite.md").read_text(encoding="utf-8"):
        raise SystemExit("import-rewrite should refuse a row written for the replaced claim:\n" + r.stdout + r.stderr)
    # The judges' sheets: one Excel sheet per judge, in the tool's order, with only the tool's texts; filled
    # in as a judge would (y, N, yes, an empty row, a note) and read back into a label file in the tool's format.
    sh(PY, ROOT / "pool" / "judges_sheet.py", "make", "--work", work, "--queries", Q, "--out", work / "sheets", "--repeat", "2")
    wb = openpyxl.load_workbook(work / "sheets" / "sheet_M.xlsx")
    ws = wb["judge"]
    hd = [c.value for c in ws[1]]
    if wb.sheetnames != ["judge"] or "GUIDELINE" not in (work / "sheets" / "HOW-TO-JUDGE_M.txt").read_text(encoding="utf-8"):
        raise SystemExit("the sheet must have one tab, 'judge', and the instructions with the guideline must be in HOW-TO-JUDGE_M.txt")
    if not ws.protection.sheet or not ws.data_validations.dataValidation \
            or any(w in str(c.value) for row in ws.iter_rows() for c in row for w in ("ranks", "boolean_formulas", "rating")):
        raise SystemExit("the judges' sheet must be protected, offer y/n only, and carry no ranks, formulas or verdicts")
    want = ["row", "query n.", "query", "background", "claim", "answer", "explanation (start)", "organisation", "country",
            "date", "note"]
    if hd[:11] != want:
        raise SystemExit(f"the sheet's columns must be {', '.join(want)}: {hd[:11]}")
    order_m = [(r[hd.index("query_id")].value, r[hd.index("doc_id")].value, r[hd.index("kind")].value) for r in ws.iter_rows(min_row=2)]
    if sum(1 for o in order_m if o[2] == "pool") != len(assign["M"]) or sum(1 for o in order_m if o[2] == "repeat") != 2:
        raise SystemExit("the sheet must hold every item of M once, plus the repeated items")
    answers = ["y", "N", "yes"]
    for i, row in enumerate(ws.iter_rows(min_row=2)):
        if i < len(order_m) - 1:  # the last row is left empty, as by a judge who stopped
            row[hd.index("answer")].value = answers[i % 3]
    ws.cell(2, hd.index("note") + 1).value = "a note"
    wb.save(work / "sheets" / "sheet_M_filled.xlsx")
    sh(PY, ROOT / "pool" / "judges_sheet.py", "read", "--work", work, "--sheet", work / "sheets" / "sheet_M_filled.xlsx", "--annotator", "M",
       "--out", work / "sheets" / "labels_M_from_sheet.jsonl")
    # a sheet made under another guideline (its recorded fingerprint differs) must be refused, not read
    wb_other = openpyxl.load_workbook(work / "sheets" / "sheet_M_filled.xlsx")
    if not str(wb_other.properties.keywords or "").startswith("guideline sha256 "):
        raise SystemExit("judges_sheet.py make should record the guideline's fingerprint in the workbook")
    wb_other.properties.keywords = "guideline sha256 " + "0" * 64
    wb_other.save(work / "sheets" / "sheet_M_other_guideline.xlsx")
    r = subprocess.run([PY, str(ROOT / "pool" / "judges_sheet.py"), "read", "--work", str(work), "--sheet",
                        str(work / "sheets" / "sheet_M_other_guideline.xlsx"), "--annotator", "M",
                        "--out", str(work / "sheets" / "labels_M_other.jsonl")], capture_output=True, text=True)
    if r.returncode == 0 or "another guideline" not in (r.stdout + r.stderr):
        raise SystemExit("a sheet made under another guideline must be refused:\n" + r.stdout + r.stderr)
    read_back = [json.loads(l) for l in open(work / "sheets" / "labels_M_from_sheet.jsonl", encoding="utf-8")]
    if len(read_back) != len(order_m) - 1 or {r["label"] for r in read_back} != {"y", "n"} or read_back[0]["note"] != "a note" \
            or any(r["seconds"] is not None for r in read_back):
        raise SystemExit(f"the sheet must read back into one label per answered row, y/n only, with the note and no time: {len(read_back)}")
    # a second round for one judge (make --only M --round 2): the same items in a fresh blank sheet with another row
    # order, the first sheet untouched; read takes it and records the sheet's file name in every label
    first_sheet = (work / "sheets" / "sheet_M.xlsx").read_bytes()
    sh(PY, ROOT / "pool" / "judges_sheet.py", "make", "--work", work, "--queries", Q, "--out", work / "sheets", "--repeat", "2",
       "--only", "M", "--round", "2")
    if (work / "sheets" / "sheet_R_round2.xlsx").exists() or not (work / "sheets" / "HOW-TO-JUDGE_M_round2.txt").exists() \
            or (work / "sheets" / "sheet_M.xlsx").read_bytes() != first_sheet:
        raise SystemExit("make --only M --round 2 must write sheet_M_round2.xlsx and its how-to only, and leave sheet_M.xlsx alone")
    wb2 = openpyxl.load_workbook(work / "sheets" / "sheet_M_round2.xlsx")
    ws2 = wb2["judge"]
    order_m2 = [(r[hd.index("query_id")].value, r[hd.index("doc_id")].value, r[hd.index("kind")].value) for r in ws2.iter_rows(min_row=2)]
    if sorted(o for o in order_m2 if o[2] == "pool") != sorted(o for o in order_m if o[2] == "pool") \
            or any(r[hd.index("answer")].value for r in ws2.iter_rows(min_row=2)):
        raise SystemExit("the second-round sheet must hold the same items as the first, with no answer")
    if order_m2 == order_m:
        raise SystemExit("the second-round sheet must have another row order")
    for row in ws2.iter_rows(min_row=2):
        row[hd.index("answer")].value = "n"
    wb2.save(work / "sheets" / "sheet_M_round2_filled.xlsx")
    sh(PY, ROOT / "pool" / "judges_sheet.py", "read", "--work", work, "--sheet", work / "sheets" / "sheet_M_round2_filled.xlsx",
       "--annotator", "M", "--out", work / "sheets" / "labels_M_round2.jsonl")
    round2 = [json.loads(l) for l in open(work / "sheets" / "labels_M_round2.jsonl", encoding="utf-8")]
    if len(round2) != len(order_m2) or {r["sheet"] for r in round2} != {"sheet_M_round2_filled.xlsx"}:
        raise SystemExit("the second-round sheet must read back whole, each label naming its sheet")
    sh(PY, ROOT / "qrels" / "build_qrels.py", "--work", work, "--annotators", "R,M",
       "--llm-labels", work / "judge" / f"labels_llm_{validation['main_judge']}.jsonl")
    # a second row of ranked runs without tests (--extra): here a copy of the BM25 run plays bm25_claimart
    shutil.copy(work / "runs" / "bm25.trec", work / "runs" / "bm25_claimart.trec")
    measured = ["--ranked", "bm25,pipeline", "--generous", "relaxed_writer_a,relaxed_writer_b", "--extra", "bm25_claimart", "--sets", sets_tags]
    for name in ("ai", "human", "ai_check"):
        sh(PY, ROOT / "eval" / "metrics.py", "--work", work, "--qrels", work / "qrels" / f"qrels_{name}.txt", "--name", name, *measured)
    sh(PY, ROOT / "eval" / "metrics.py", "--work", work, "--qrels", work / "qrels" / "qrels_ai.txt", "--name", "ai_before", "--before", "2023-01", *measured)
    mb = json.loads((work / "eval" / "metrics_ai_before.json").read_text(encoding="utf-8"))
    for s in ("bm25", "pipeline", "relaxed_writer_a", "relaxed_writer_b"):
        if mb["summary"][s]["judged@20"] != 1.0:
            raise SystemExit(f"after extend_pool_before.py every document of the top 20 before 2023-01 must be judged: {s} {mb['summary'][s]['judged@20']}")
    ma = json.loads((work / "eval" / "metrics_ai.json").read_text(encoding="utf-8"))
    if "bm25_claimart" not in (work / "eval" / "metrics_ai.md").read_text(encoding="utf-8"):
        raise SystemExit("metrics.py --extra must report the second-row run bm25_claimart")
    for w in sets_tags.split(","):
        for q, v in ma["per_query"][w].items():
            if v["B"] == 0 and v["set_F1"] != 0.0:
                raise SystemExit(f"an empty set must count as F1 0 (it was left out of the mean before): {w} {q}")
    # A set larger than the run is compared at the run's length: 2 relevant among the 3 documents returned
    # gives P@B 2/3, not 2/5.
    sys.path.insert(0, str(ROOT / "eval"))
    import metrics as mt
    v = mt.at_B({"q": [("a", 3.0), ("b", 2.0), ("c", 1.0)]}, {"q": {"a": 1, "c": 1, "z": 1}}, {"q": 5}, ["q"])["q"]
    if abs(v["P@B"] - 2 / 3) > 1e-9 or abs(v["R@B"] - 2 / 3) > 1e-9 or v["note"] != "B>run length":
        raise SystemExit(f"a list must be compared at its own length when the set is larger: {v}")
    for w, wname in (("graph_writer_a_claim", "ai_a"), ("graph_writer_b_claim", "ai_b")):
        sh(PY, ROOT / "eval" / "failure_protocol.py", "--work", work, "--queries", Q, "--qrels", work / "qrels" / "qrels_ai.txt", "--graph-tag", w, "--name", wname, "--node", "claim")
        sh(PY, ROOT / "eval" / "oracle_formula.py", "--work", work, "--qrels", work / "qrels" / "qrels_ai.txt", "--graph-tag", w, "--name", wname, "--node", "claim")
    sh(PY, ROOT / "eval" / "writer_agreement.py", "--work", work, "--queries", Q, "--authors", "writer_a,writer_b", "--node", "claim", "--qrels", work / "qrels" / "qrels_ai.txt",
       "--tex", work / "figures" / "writer_numbers.tex")
    wn = (work / "figures" / "writer_numbers.tex").read_text(encoding="utf-8")
    if "\\newcommand{\\writerAFormulas}" not in wn or "\\newcommand{\\writersPagesJaccard}" not in wn or any(ch.isdigit() for ch in "".join(
            line.split("{")[1] for line in wn.splitlines() if line.startswith("\\newcommand"))):
        raise SystemExit("writer_agreement.py --tex must write the writers' macros, with names LaTeX accepts (no digits)")
    sh(PY, ROOT / "stats" / "archive_stats.py", "--store", toy / "store", "--work", work)
    sh(PY, ROOT / "snapshot" / "publisher_map.py", "--work", work, "--store", toy / "store", "--ifcn-sources", toy / "ifcn_fake.json")
    sh(PY, ROOT / "stats" / "archive_stats.py", "--store", toy / "store", "--work", work, "--country-map", work / "stats" / "publishers_map.csv")
    # The language rule alone on the corpus file (build_store.py above already applied it while building store2).
    sh(PY, ROOT / "snapshot" / "language_rule.py", "--corpus", work / "corpus.jsonl", "--out", work / "stats" / "language_decisions.jsonl")
    sh(PY, ROOT / "stats" / "plot_archive.py", "--work", work, "--store", toy / "store", "--out", work / "figures")
    # The paper's other own facts (paper_facts.py) on the toy snapshot: it needs the mirror's manifest, which the toy
    # mirror (dumped above, not downloaded) does not have, so a minimal one is written. On toy data the claims about
    # the real archive (India the largest country, Mars ...) fail by design, hence --allow-fail.
    n_lines = sum(1 for line in open(toy / "toy_mirror.nt", encoding="utf-8") if line.strip())
    (toy / "toy_mirror_manifest.json").write_text(json.dumps({"date": "2026-10-02T00:00:00+00:00", "triples": n_lines,
                                                               "n_fact_checks_listed": json.loads((toy / "snapshot_manifest.json").read_text(encoding="utf-8"))["claimreviews_loaded"]}),
                                                   encoding="utf-8")
    sh(PY, ROOT / "stats" / "paper_facts.py", "--work", work, "--store", toy / "store2", "--mirror", toy / "toy_mirror.nt",
       "--mirror-manifest", toy / "toy_mirror_manifest.json", "--runs", work / "runs" / "manifest.json", "--queries", Q,
       "--pilot", work / "queries_pilot.json", "--out", work / "figures", "--allow-fail")
    sh(PY, ROOT / "model_selection" / "pareto_simple.py", "--out-report", work / "model_report.md", "--out-tex", work / "figures" / "model_numbers.tex")
    sh(PY, ROOT / "queries" / "allocate_queries.py", "--stats", work / "stats", "--total", "20", "--floor", "2", "--since", "2016", "--min-docs", "1",
       "--out-md", work / "stats" / "query_allocation.md", "--out-tex", work / "figures" / "allocation.tex")
    # The writers' tools: the entity catalog from the toy entity index (claim node, which run_all built above), a search,
    # a sheet template, an import of a filled sheet (one title deliberately absent from the archive), and a graph run on the result.
    cat = work / "stats" / "entity_catalog.csv"
    sh(PY, ROOT / "queries" / "entity_catalog.py", "build", "--work", work, "--node", "claim")
    sh(PY, ROOT / "queries" / "entity_catalog.py", "search", "--catalog", cat, "vacc", "--top", "5")
    sh(PY, ROOT / "queries" / "entity_catalog.py", "exact", "--catalog", cat, "Vaccine", "Not a page")
    sh(PY, ROOT / "queries" / "import_formulas.py", "template", "--queries", Q, "--out", work / "sheet_writer_c_template.xlsx", "--author", "writer_c")
    sh(PY, ROOT / "queries" / "import_formulas.py", "template", "--queries", Q, "--out", work / "sheet_half1.csv", "--author", "writer_a", "--half", "1of2", "--common", "1")
    sh(PY, ROOT / "queries" / "import_formulas.py", "template", "--queries", Q, "--out", work / "sheet_all_ordered.csv", "--author", "writer_a", "--half", "1of2", "--common", "1", "--all-ordered")
    # A filled sheet as Excel: the template read back, two rows filled, saved again (what a writer does). Titles in quotes
    # (straight and the curly ones Excel produces); 'Not a page' is deliberately absent from the archive.
    import openpyxl
    wb = openpyxl.load_workbook(work / "sheet_writer_c_template.xlsx")
    ws = wb.active
    filled = {"q1": ('"Vaccine" AND ("Microchip" OR “Not a page”)', "activated by 5G", "7,5", "2", "toy row"),
              "q2": ('"Ukraine" AND "Biological weapon" AND NOT "NATO"', "", "12 min", "0", "")}  # minutes as a writer types them
    for row in ws.iter_rows(min_row=2):
        if row[0].value in filled:
            for cell, val in zip(row[3:8], filled[row[0].value]):
                cell.value = val
    wb.save(work / "sheet_writer_c.xlsx")
    # The older key form (formula with keys plus an 'entities' column) as a CSV, still accepted.
    (work / "sheet_writer_d.csv").write_text(
        "query_id,author_id,narrative,formula,entities,not_expressible,minutes,refinements,notes\n"
        "q1,writer_d,edited narrative,e1 AND e2,e1 = Vaccine; e2 = Microchip,,5,1,key form\n", encoding="utf-8-sig")
    # A redirect cache written by hand (the toy never calls Wikipedia): 'Biological weapon' is a redirect to 'Biological agent'.
    (work / "wikipedia_redirects.json").write_text(json.dumps({"Biological weapon": {"resolved": "Biological agent", "exists": True}}), encoding="utf-8")
    # The other direction: names that redirect TO a page today. The toy plays 'Kyiv' and 'Ukrainian SSR' as
    # older names of 'Ukraine'; only 'Kyiv' is in the toy archive, so only it may enter the formula.
    (work / "wikipedia_incoming_redirects.json").write_text(json.dumps({"Ukraine": ["Kyiv", "Ukrainian SSR"]}), encoding="utf-8")
    sh(PY, ROOT / "queries" / "import_formulas.py", "import", "--queries", Q, "--sheet", work / "sheet_writer_c.xlsx", "--catalog", cat,
       "--redirect-cache", work / "wikipedia_redirects.json", "--exclusions", work / "exclusions_writers.json", "--exclude-as", "C",
       "--out", work / "queries_imported.json")
    sh(PY, ROOT / "queries" / "import_formulas.py", "import", "--queries", work / "queries_imported.json", "--sheet", work / "sheet_writer_d.csv",
       "--out", work / "queries_imported.json")
    imported = {q["query_id"]: q for q in json.loads((work / "queries_imported.json").read_text(encoding="utf-8"))}
    mins = {f["author_id"]: f.get("minutes") for f in imported["q2"]["boolean_formulas"]}
    # The store's spelling of a page (--catalog): the exact title first; a page that differs only by capital letters
    # only when the exact one is not in the store, the one with the most documents (the real store
    # holds 53 such pairs, e.g. Islamic_State 738 documents and Islamic_state 6).
    sys.path.insert(0, str(ROOT))
    from queries.import_formulas import CatalogLookup
    from queries.entity_catalog import norm_key, uri_from_title
    fake = [{"dbpedia_uri": "http://dbpedia.org/resource/" + n, "n_docs": c} for n, c in
            (("Islamic_State", 738), ("Islamic_state", 6), ("CRIME", 9), ("Crime_scene", 4), ("QI", 7), ("Qi", 73))]
    for r in fake:
        r["_key"] = norm_key(r["dbpedia_uri"])
    look = CatalogLookup(fake)
    got = {t: (look.find(uri_from_title(t))[0] or {}).get("dbpedia_uri", "").rsplit("/", 1)[-1] for t in ("Islamic state", "Islamic State", "Crime", "QI", "Qi", "Nothing")}
    if got != {"Islamic state": "Islamic_state", "Islamic State": "Islamic_State", "Crime": "CRIME", "QI": "QI", "Qi": "Qi", "Nothing": ""}:
        raise SystemExit(f"the catalog must prefer the exact page and use a capital-letter match only when the exact page is missing: {got}")
    # A title with a slash is one title ('HIV/AIDS'), not the part after the slash .
    from queries.entity_catalog import title_from_uri
    slash = [{"dbpedia_uri": "http://dbpedia.org/resource/HIV/AIDS", "n_docs": 5}, {"dbpedia_uri": "http://dbpedia.org/resource/India", "n_docs": 50}]
    for r in slash:
        r["_key"] = norm_key(r["dbpedia_uri"])
    look2 = CatalogLookup(slash)
    if title_from_uri("http://dbpedia.org/resource/HIV/AIDS") != "HIV/AIDS" or (look2.find(uri_from_title("Mass media/India"))[0] is not None) \
            or (look2.find(uri_from_title("HIV/AIDS"))[0] or {}).get("n_docs") != 5:
        raise SystemExit("a title with a slash must be read whole: 'HIV/AIDS' is one page and 'Mass media/India' is not 'India'")
    if mins.get("writer_c") != 12:
        raise SystemExit(f"import should read the minutes '12 min' as 12, got {mins.get('writer_c')!r}")
    fc = next(f for f in imported["q2"]["boolean_formulas"] if f["author_id"] == "writer_c")
    if not any(e.get("alias_of") for e in fc["entities"].values()) or "w)" not in fc["formula"]:
        raise SystemExit("import should keep 'Biological weapon' next to its redirect 'Biological agent' (rule 'either page'): " + fc["formula"])
    older = [e["label"] for e in fc["entities"].values() if e.get("alias_kind") == "older name"]
    if older != ["Kyiv"] or "o1)" not in fc["formula"]:
        raise SystemExit(f"import should add the older name Kyiv of Ukraine (in the archive) and not Ukrainian SSR (not in it): {older} {fc['formula']}")
    # The writers' feedback file (queries/title_check.py), offline with hand-written caches: 'Microchip' plays a
    # disambiguation page, 'Not a page' a missing page; the redirect of 'Biological weapon' is followed before the check.
    (work / "tc_redirects.json").write_text(json.dumps({
        "Vaccine": {"resolved": "Vaccine", "exists": True}, "Microchip": {"resolved": "Microchip", "exists": True},
        "Not a page": {"resolved": "Not a page", "exists": False}, "Ukraine": {"resolved": "Ukraine", "exists": True},
        "Biological weapon": {"resolved": "Biological agent", "exists": True}, "NATO": {"resolved": "NATO", "exists": True}}), encoding="utf-8")
    (work / "tc_disambiguation.json").write_text(json.dumps({"Vaccine": False, "Microchip": True, "Not a page": False, "Ukraine": False,
                                                             "Biological agent": False, "NATO": False}), encoding="utf-8")
    sh(PY, ROOT / "queries" / "title_check.py", "--sheet", work / "sheet_writer_c.xlsx", "--writer-name", "Toy", "--out", work / "title_check_toy.txt",
       "--csv", work / "title_check_toy.csv", "--redirect-cache", work / "tc_redirects.json", "--disambiguation-cache", work / "tc_disambiguation.json", "--offline")
    tc = (work / "title_check_toy.txt").read_text(encoding="utf-8")
    if '"Microchip" è una pagina di disambiguazione' not in tc or '"Not a page" non è una pagina' not in tc or "Biological" in tc:
        raise SystemExit("title_check should flag Microchip (A) and Not a page (B), and nothing else: " + tc)
    # The main formula of every query is the assigned writer's; writer_c and writer_d stand in for the two writers (exit 1 is
    # expected when a writer has no formula for a query yet, which is the case in the toy: the run checks the file, not the code).
    r = subprocess.run([str(PY), str(ROOT / "queries" / "import_formulas.py"), "assign-main", "--queries", str(work / "queries_imported.json"),
                        "--out", str(work / "queries_imported.json"), "--authors", "writer_c,writer_d"], cwd=ROOT)
    assert r.returncode in (0, 1), "assign-main crashed"
    sh(PY, ROOT / "systems" / "graph_boolean.py", "--queries", work / "queries_imported.json", "--store", toy / "store", "--work", work,
       "--author", "writer_c", "--backend", "index", "--node", "claim")
    # The same formulas with today's pages only (no other names), and the cost of the old names (eval/stale_names.py) on a
    # hand-made qrels where every document of the main sets is relevant: what is lost is exactly the difference of the sets.
    sh(PY, ROOT / "systems" / "graph_boolean.py", "--queries", work / "queries_imported.json", "--store", toy / "store", "--work", work,
       "--author", "writer_c", "--backend", "index", "--current-names-only", "--node", "claim")
    main_sets = {r["query_id"]: set(r["doc_ids"]) for r in map(json.loads, open(work / "runs" / "graph_writer_c_claim_sets.jsonl", encoding="utf-8"))}
    cur_sets = {r["query_id"]: set(r["doc_ids"]) for r in map(json.loads, open(work / "runs" / "graph_writer_c_claim_current_sets.jsonl", encoding="utf-8"))}
    if any(not cur_sets[q] <= main_sets[q] for q in main_sets):
        raise SystemExit("today's pages only must give a subset of the main set")
    with open(work / "qrels_stale_toy.txt", "w", encoding="utf-8") as fh:
        for q, ds in main_sets.items():
            for d in sorted(ds):
                fh.write(f"{q} 0 {d} 1\n")
    sh(PY, ROOT / "eval" / "stale_names.py", "--work", work, "--qrels", work / "qrels_stale_toy.txt", "--name", "toy", "--tags", "graph_writer_c_claim")
    st = json.loads((work / "eval" / "stale_names_toy.json").read_text(encoding="utf-8"))["writers"]["graph_writer_c_claim"]
    if st["relevant_lost"] != sum(len(main_sets[q] - cur_sets[q]) for q in main_sets):
        raise SystemExit(f"stale_names should count as lost exactly the relevant documents only the other names reach: {st['relevant_lost']}")
    # a writers' folder as the release finds it: the toy's two Wikipedia caches and a repair log with a column that must not be released
    (work / "writers_toy").mkdir(exist_ok=True)
    for name in ("wikipedia_redirects.json", "wikipedia_incoming_redirects.json"):
        shutil.copy2(work / name, work / "writers_toy" / name)
    (work / "writers_toy" / "repair_log.csv").write_text("query_id,writer,sheet_line,original,repaired,rule,reason_it,done_by,date,confirm_with_writer\n"
                                                        "q1,writer_c,2,a AND AND b,a AND b,doubled AND,nota,Name,2026-11-17,no\n", encoding="utf-8-sig")
    sh(PY, ROOT / "release" / "export_release.py", "--work", work, "--queries", Q, "--out", work / "release_build", "--writers", work / "writers_toy")
    rel_readme = (work / "release_build" / "README.md").read_text(encoding="utf-8")
    rel_q = json.loads((work / "release_build" / "queries.json").read_text(encoding="utf-8"))
    if any("passage" in (q.get("provenance") or {}) or "rewrite_context" in (q.get("provenance") or {})
           or {"context", "note"} & set(((q.get("provenance") or {}).get("rewrite_closed") or {})) for q in rel_q):
        raise SystemExit("the released query file must not quote the reports (passage, context)")
    rel_pool = (work / "release_build" / "pool" / "pool.jsonl").read_text(encoding="utf-8").splitlines()
    rel_labels = list((work / "release_build" / "labels").glob("labels_*.jsonl"))
    if not rel_pool or "claim_text" in rel_pool[0] or not rel_labels or any('"note"' in p.read_text(encoding="utf-8") for p in rel_labels):
        raise SystemExit("the release must carry the pool without texts and the human label files without notes")
    if "local only" in rel_readme or "## Plain-language documentation" in rel_readme:
        raise SystemExit("the release README must not carry the section marked 'local only'")
    rel = work / "release_build" / "writers"
    if not (rel / "wikipedia_incoming_redirects.json").exists() or "done_by" in (rel / "formula_repairs.csv").read_text(encoding="utf-8") \
            or "a AND AND b" not in (rel / "formula_repairs.csv").read_text(encoding="utf-8"):
        raise SystemExit("the release must carry the Wikipedia caches and the repairs, without the names column")
    # The Results section's numbers, tables and figures (eval/paper_results.py), read from the evaluation files above.
    # The toy has no table per single model judge, so those rows are left out and their numbers print as "--".
    ex_q = next(q for q, n in ma["n_relevant_per_query"].items() if n > 0)
    sh(PY, ROOT / "eval" / "paper_results.py", "--work", work, "--queries", Q, "--out", work / "figures",
       "--generous", "relaxed_writer_a,relaxed_writer_b", "--extra", "bm25_claimart", "--sets", sets_tags,
       "--judges", ",".join(c for c in validation["candidates"] if c != "majority"), "--stale-name", "toy",
       "--writers-name", "", "--example-query", ex_q)
    rn = (work / "figures" / "results_numbers.tex").read_text(encoding="utf-8")
    if "\\newcommand{\\resBMRTwenty}" not in rn or "\\newcommand{\\resOracleA}" not in rn or any(ch.isdigit() for ch in "".join(
            line.split("{")[1] for line in rn.splitlines() if line.startswith("\\newcommand"))):
        raise SystemExit("paper_results.py must write the Results macros, with names LaTeX accepts (no digits)")
    # The oracle ceiling on a worked example (eval/oracle_formula.py).
    sys.path.insert(0, str(ROOT / "eval"))
    import oracle_formula as orc
    cands, rel, judged = {"a": {"1", "2", "3", "4"}, "b": {"5", "6"}, "c": {"1", "7", "8", "9", "10"}}, {"1", "2", "5"}, {"1", "2", "3", "5", "6", "7"}
    g = orc.greedy(cands, rel, judged, 10, "strict")
    if g["terms"] != ["a", "b"] or abs(g["curve"][-1] - 2 / 3) > 1e-9:
        raise SystemExit(f"oracle greedy (strict) should take a then b, F1 0.667: {g['terms']} {g['curve'][-1]}")
    g = orc.greedy(cands, rel, judged, 10, "judged_only")
    if g["terms"] != ["a", "b"] or abs(g["curve"][-1] - 0.75) > 1e-9:
        raise SystemExit(f"oracle greedy (judged_only) should end at F1 0.75: {g['terms']} {g['curve'][-1]}")
    bc = orc.best_cutoff(["1", "3", "2", "9", "5"], rel, judged, "strict")
    if bc["k"] != 5 or abs(bc["F1"] - 0.75) > 1e-9:
        raise SystemExit(f"oracle best cutoff should be k 5, F1 0.75: {bc}")
    expected = ["runs/graph.trec", "runs/bm25.trec", "runs/pipeline.trec", "runs/dense.trec", "runs/fused.trec", "runs/manifest.json",
                "pool/pool.jsonl", "labels/agreement.md", "judge/validation.md", "qrels/qrels_human.txt", "qrels/qrels_ai.txt",
                "eval/metrics_ai.md", "eval/failures_ai_a.csv", "eval/oracle_ai_a.md", "eval/writer_agreement.md", "stats/archive_stats.md", "release_build/CHECKSUMS.sha256",
                "stats/publishers_map.csv", "../snapshot_manifest.json", "../language_decisions.jsonl", "stats/language_decisions.jsonl",
                "stats/country_month.csv", "stats/publishers.csv", "figures/numbers.tex", "figures/per_month.pdf", "stats/query_allocation.md",
                "figures/facts_numbers.tex", "stats/paper_facts.md", "figures/model_numbers.tex",
                "stats/entity_catalog.csv", "queries_imported.json",
                "drafting/candidates_us-2020.jsonl", "drafting/merged_us-2020.jsonl", "drafting/reviewed.jsonl",
                "drafting/review_agreement.md", "queries_drawn.json", "queries_pilot.json", "queries_v7.json",
                "drafting/rewrite_proposals_queries_drawn.jsonl",
                "judge_bundle/MANIFEST.json", "judge_bundle.zip", "judge_bundle/judge/labels_llm_fakerun.jsonl", "judge_bundle/judge/run_fakerun.json",
                "judge_bundle/judge/labels_llm_fakeapi.jsonl", "judge_bundle/judge/run_fakeapi.json",
                "pool/check.json", "judge/labels_llm_majority.jsonl", "judge_results.zip", "eval/stale_names_toy.md",
                "figures/results_numbers.tex", "figures/results_ranked.tex", "figures/results_recall.pdf"]
    if (ROOT / "queries" / "stability_needs.py").exists():
        expected.append("stability/stability.md")
    missing = [e for e in expected if not (work / e).exists()]
    if missing:
        raise SystemExit(f"MISSING OUTPUTS: {missing}")
    print("\nTOY RUN OK: every script ran and every expected file exists.")
    if not args.keep:
        shutil.rmtree(toy)


if __name__ == "__main__":
    main()
