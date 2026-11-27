# The AI judge

Open-weight language models label every pooled (query, document) pair as relevant or not, zero-shot, with one fixed
prompt (`prompts.py`): the human judges' guideline (`pool/GUIDELINE.md`) in the shape of the UMBRELA prompt, the same
text the human judges see, answer YES or NO. Two human judges label a check sample of 25 queries
(`pool/build_pool.py`, `pool/judges_sheet.py`), which measures how good each AI judge is. No human label goes into the
prompt, so the AI run and the human check can happen at the same time. Every script talks to a server through the
OpenAI chat API (`llm_client.py`). Every label records the model, the precision, the server, the prompt's sha256
fingerprint (no version number), the token counts and, through an API that routes to several providers, the provider
that answered and the cost. The models are chosen by section 4 of `model_selection/RULES.md` (version 7):
gpt-oss-120b (OpenAI), Llama-3.3-70B-Instruct (Meta) and DeepSeek-V3.2 (DeepSeek), served through OpenRouter in a
declared precision by providers fixed in `judges_api.json`; no Qwen model, because the pipeline's models are Qwen3.

## The steps

1. Once the pool is built: `python judge/pack_for_judge.py --work WORK --queries Q.json --out judge_bundle --zip`. The
   bundle holds only what the prompt uses: no verdict, no system names, no human label. It is data, so it never goes
   into the repository.
2. The labels (`RUN-WITH-API.md`): `run_judges.py run --config judge/judges_api.json --bundle judge_bundle`, first with
   `--max-items 200` to measure the cost. (`RUN-ON-GPU.md` runs open models on one's own GPUs with vLLM instead.)
3. Copy the bundle's `judge/labels_llm_*.jsonl` into `WORK/judge/`, then, after the human check, run
   `adjudicate.py`, `validate_judge.py` and `qrels/build_qrels.py --llm-labels <the main judge's file>`.

Models that reason before answering (gpt-oss) need room for the reasoning: `max_tokens` about 4096 and, through
OpenRouter, `extra` `{"reasoning": {"effort": "medium"}}` (through vLLM, `{"reasoning_effort": "medium"}`); a reply cut
by the limit is not written and is asked again on the next run.

## Choosing the main judge

1. `adjudicate.py --annotators R,M` gives the two human judges' kappa on the 150 documents both judged. There is no
   discussion: the human label of every checked document is the label of the query's owner, the judge who reads the
   whole query (TREC's primary assessor).
2. `validate_judge.py --annotators R,M` adds the majority of the three models as a fourth candidate and applies the
   pass rule: kappa with each human judge at least the lower end of the 95 percent bootstrap interval of the two
   judges' kappa, and the systems in the same order under human and AI labels on the check queries. The main judge is
   the passing candidate with the highest mean kappa; if none passes, the best is used and the paper says so.
3. `qrels/build_qrels.py --llm-labels <the main judge's file>`: the AI labels for every pair (the main tables) and the
   human labels of the check sample as a separate table.
