#!/usr/bin/env bash
# reproduce.sh -- Linux reproduction of the automatic steps: the runs of the three systems and the pool.
# Needs: the English CimpleKG Oxigraph store (STORE), the query file with both writers' formulas (Q), a GPU (or the
# cpu-test profile for a test). The numbers of Sections 3 to 5 are checked separately: reproduce_paper.py.
# Usage: STORE=/data/snapshot-2026-10/final/store Q=queries_with_formulas.json PROFILE=laptop ./reproduce.sh
set -euo pipefail
STORE="${STORE:?set STORE to the Oxigraph store directory}"
Q="${Q:?set Q to the query file with the formulas of both writers}"
PROFILE="${PROFILE:-laptop}"            # laptop (the paper's run) | cpu-test (code test only)
WORK="${WORK:-work}"
python -m pip install -r requirements.txt
# The second row first: the text systems on the claim followed by the explanation, renamed *_claimart.
python run_all.py --queries "$Q" --store "$STORE" --work "$WORK" --profile "$PROFILE" --node cr --text-mode claim+article \
  --steps prepare,bm25,embed,retrieve,rerank,manifest
for f in bm25 dense fused pipeline; do mv "$WORK/runs/$f.trec" "$WORK/runs/${f}_claimart.trec"; done
mv "$WORK/runs/manifest.json" "$WORK/runs/manifest_claimart.json"
rm -rf "$WORK/bm25_index" "$WORK/dense_$PROFILE"
# The main runs: the explanation alone, the text the graph's entities come from.
python run_all.py --queries "$Q" --store "$STORE" --work "$WORK" --profile "$PROFILE" --node cr --text-mode article \
  --steps bm25,embed,retrieve,rerank,manifest
for A in annotator_a annotator_b; do
  python systems/graph_boolean.py --queries "$Q" --store "$STORE" --work "$WORK" --author "$A" --node cr --backend index
  python systems/graph_boolean.py --queries "$Q" --store "$STORE" --work "$WORK" --author "$A" --node cr --backend index --current-names-only
  python systems/ranked_entities.py --queries "$Q" --work "$WORK" --node cr --author "$A" --tag "relaxed_$A"
done
# The fact-checkers' countries, shown to the judges and in the AI prompt (the IFCN list of the snapshot; by default
# next to the snapshot folder that holds the store).
IFCN="${IFCN:-$(dirname "$(dirname "$STORE")")/ifcn_sources.json}"
python snapshot/publisher_map.py --work "$WORK" --store "$STORE" --ifcn-sources "$IFCN" --out "$WORK/stats/publishers_map.csv"
python pool/build_pool.py --work "$WORK" --queries "$Q" --ranked bm25,pipeline --sets graph_annotator_a_cr,graph_annotator_b_cr \
  --ai-only relaxed_annotator_a,relaxed_annotator_b,bm25_claimart,pipeline_claimart --judges R,M --country-map "$WORK/stats/publishers_map.csv"
python judge/pack_for_judge.py --work "$WORK" --queries "$Q" --out "$WORK/judge_bundle" --zip
echo "Now: the AI judges through the API (judge/RUN-WITH-API.md; on GPUs: judge/RUN-ON-GPU.md); pool/judges_sheet.py make, then read, for each human judge; judge/adjudicate.py;"
echo "judge/validate_judge.py; qrels/build_qrels.py; eval/metrics.py, eval/failure_protocol.py, eval/oracle_formula.py,"
echo "eval/stale_names.py, eval/writer_agreement.py (metrics.py with --extra bm25_claimart,pipeline_claimart); release/export_release.py --writers DIR (README.md, the steps in order). Code check without data: python tests/run_toy.py"
