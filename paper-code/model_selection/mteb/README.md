# The leaderboard files

`english/MTEB_eng_v2_summary(1).csv` and `english/MTEB_eng_v2_per_task(1).csv` are the MTEB(eng, v2) leaderboard as
downloaded on 17 September 2026 from the MTEB leaderboard (https://huggingface.co/spaces/mteb/leaderboard), unchanged.
`model_selection/pareto_simple.py` applies rule S of `model_selection/RULES.md` to the summary file.

The scores come from the MTEB results repository (https://github.com/embeddings-benchmark/results, CC0 1.0 since
5 August 2026; the Hugging Face dataset mteb/results is CC0 1.0 as well). The model metadata (licence, size, memory,
the training data declared) come from the mteb package (https://github.com/embeddings-benchmark/mteb, Apache License
2.0).
