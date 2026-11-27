# reproduce.ps1 -- Windows reproduction of the automatic steps: the runs of the three systems and the pool.
# Usage (PowerShell):  $env:STORE="C:\snapshot-2026-10\final\store"; $env:WORK="C:\snapshot-2026-10\work"; $env:Q="queries_with_formulas.json"; .\reproduce.ps1
# The numbers of Sections 3 to 5 are checked separately: reproduce_paper.py.
$ErrorActionPreference = "Stop"
if (-not $env:STORE) { throw "set STORE to the Oxigraph store directory" }
if (-not $env:Q) { throw "set Q to the query file with both writers' formulas" }
$WORK = if ($env:WORK) { $env:WORK } else { "work" }
$Q = $env:Q
$PROFILE_NAME = if ($env:PROFILE_NAME) { $env:PROFILE_NAME } else { "laptop" }
python -m pip install -r requirements.txt
# The second row first: the text systems on the claim followed by the explanation, renamed *_claimart.
python run_all.py --queries $Q --store $env:STORE --work $WORK --profile $PROFILE_NAME --node cr --text-mode claim+article --steps prepare,bm25,embed,retrieve,rerank,manifest
foreach ($f in "bm25","dense","fused","pipeline") { Move-Item -Force "$WORK\runs\$f.trec" "$WORK\runs\${f}_claimart.trec" }
Move-Item -Force "$WORK\runs\manifest.json" "$WORK\runs\manifest_claimart.json"
Remove-Item -Recurse -Force "$WORK\bm25_index", "$WORK\dense_$PROFILE_NAME"
# The main runs: the explanation alone, the text the graph's entities come from.
python run_all.py --queries $Q --store $env:STORE --work $WORK --profile $PROFILE_NAME --node cr --text-mode article --steps bm25,embed,retrieve,rerank,manifest
foreach ($A in @("annotator_a", "annotator_b")) {
  python systems\graph_boolean.py --queries $Q --store $env:STORE --work $WORK --author $A --node cr --backend index
  python systems\graph_boolean.py --queries $Q --store $env:STORE --work $WORK --author $A --node cr --backend index --current-names-only
  python systems\ranked_entities.py --queries $Q --work $WORK --node cr --author $A --tag "relaxed_$A"
}
# The fact-checkers' countries, shown to the judges and in the AI prompt (the IFCN list of the snapshot; by default
# next to the snapshot folder that holds the store, as in C:\snapshot-2026-10\ifcn_sources.json).
$IFCN = if ($env:IFCN) { $env:IFCN } else { Join-Path (Split-Path (Split-Path $env:STORE)) "ifcn_sources.json" }
python snapshot\publisher_map.py --work $WORK --store $env:STORE --ifcn-sources $IFCN --out "$WORK\stats\publishers_map.csv"
python pool\build_pool.py --work $WORK --queries $Q --ranked bm25,pipeline --sets graph_annotator_a_cr,graph_annotator_b_cr --ai-only relaxed_annotator_a,relaxed_annotator_b,bm25_claimart,pipeline_claimart --judges R,M --country-map "$WORK\stats\publishers_map.csv"
python judge\pack_for_judge.py --work $WORK --queries $Q --out "$WORK\judge_bundle" --zip
Write-Host "Now: the AI judges through the API (judge\RUN-WITH-API.md; on GPUs: judge\RUN-ON-GPU.md); pool\judges_sheet.py make, then read, for each human judge; judge\adjudicate.py; judge\validate_judge.py; qrels\build_qrels.py; eval\metrics.py (with --extra bm25_claimart,pipeline_claimart), eval\failure_protocol.py, eval\oracle_formula.py, eval\stale_names.py, eval\writer_agreement.py; release\export_release.py --writers DIR (README.md, the steps in order)"
