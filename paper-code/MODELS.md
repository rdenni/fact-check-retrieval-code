# The models of the text pipeline

Chosen by the rules in `model_selection/RULES.md`: the embedder by rule S (script `model_selection/pareto_simple.py`,
report `model_selection/report-2026-09-17-v5-simple.md`), the reranker as the Qwen3-Reranker of the same size, released
with the embedder. The revisions are the Hugging Face commit hashes of the models used. Both models are Apache 2.0;
`systems/dense.py` and `systems/rerank.py` compute the embedding and the score as the examples of their model cards
do (the last-token pooling, the reranker's prompt and its yes/no score), which are under the same licence.

| role | model and revision | setting |
|---|---|---|
| embedder | Qwen/Qwen3-Embedding-0.6B @ 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 | bfloat16, sdpa attention, up to 512 tokens, last-token pooling, L2-normalised |
| reranker | Qwen/Qwen3-Reranker-0.6B @ e61197ed45024b0ed8a2d74b80b4d909f1255473 | bfloat16, sdpa attention, up to 512 tokens, score = P(yes) from the yes/no logits |
| BM25 | bm25s, Lucene scoring | k1 = 0.9, b = 0.4, English stop words, Snowball stemmer (the BEIR reference setting) |
| fusion | reciprocal rank fusion | k = 60, two channels of depth 100 |

Query instruction (embedder and reranker): "Given a description of a disinformation narrative, retrieve fact-checked
claims that are instances or variants of it". Documents carry no instruction.

Hardware of the paper's runs: an NVIDIA RTX 4070 Laptop GPU with 8 GB of memory (`profiles.py`, profile `laptop`).
The 0.6B models take about 1.2 GB each in bfloat16; the embedder and the reranker are never loaded together. The
measured speed is in `WORK/dense_laptop/meta.json` and `WORK/rerank_laptop_meta.json`, the GPU name in
`runs/manifest.json`.

The AI judges are chosen by section 4 of `model_selection/RULES.md` (version 7): gpt-oss-120b, Llama-3.3-70B-Instruct
and DeepSeek-V3.2, run through OpenRouter. Their precision and the providers allowed are in `judge/judges_api.json`;
every label records the provider that answered.

Licences of the judges. gpt-oss-120b is under the Apache 2.0 licence and DeepSeek-V3.2 under the MIT licence.
Llama-3.3-70B-Instruct is under the Llama 3.3 Community License. Built with Llama. Llama 3.3 is licensed under the
Llama 3.3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. The labels it produced are
released with the collection. Under section 1.b.i of that licence, a model that is trained or improved with outputs
of Llama 3.3 and then distributed must carry "Llama" at the start of its name.
