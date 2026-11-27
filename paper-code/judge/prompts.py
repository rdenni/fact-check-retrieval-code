"""
prompts.py -- the AI judge's prompt (model_selection/RULES.md, section 4). Fixed before any label and never tuned: changing a word changes the labels (Thomas et al. 2024).
The text is the human judges' guideline (pool/GUIDELINE.md), the same task, the same rules and the same examples, with
two differences that only concern the form of the answer: the AI answers YES or NO (the humans y or n), and an
unclear claim is answered NO without the note that the humans write. prompt_fingerprint() identifies the exact text:
every label records it.

What the AI reads, the same as the human judges see in pool/judges_sheet.py: the query, its background if any, the
claim, the first 500 characters of the explanation (the fact-checker's text; pool/build_pool.py --context-chars), the
fact-checker, its country and the date. It never sees the fact-checker's verdict, nor which system returned the
document. The answer is one word, YES or NO.

Its shape is UMBRELA's (instructions, the inputs, a fixed answer format; Upadhyay et al. 2024, Figure 2), not its
definition: not its 0-3 scale (a yes/no label, as the human judges give) and not its step that asks how trustworthy the
passage is (against the rule that the verdict never matters). TREC 2025 RAG did the same: it put its own assessors'
definition into the UMBRELA shape. The judge may use its general knowledge, as a human judge does (who a famous person
is); UMBRELA's prompt forbids nothing of the kind either.
"""
from __future__ import annotations

import hashlib

_GUIDE = """What this is about
A fact-checker is an expert who checks whether claims are true. For each claim, the fact-checker publishes a verdict (for example true, false, or something in between) and an explanation of that verdict.

You are a misinformation researcher. You are looking for claims that have been fact-checked. A query describes what you are looking for.

What you see
- the query.
- the background (only for some queries): one sentence that helps you understand the query when the query alone may not be clear. It is not part of the query.
- the claim.
- the start of the explanation (its first 500 characters).
- the organisation the fact-checker works for, its country, and the date.

Your task
Read the query and the claim. Answer YES if the claim does at least one of these things:
- The claim contains what the query says. The words can be the same or different.
- The claim asks whether what the query says is true.
- The claim says that what the query says is false.
- The claim describes one specific case of what the query says. This applies only when the query is general. When the query is about one specific event, the claim must be about that same event.
- The claim tries to prove what the query says, for example with a photo, a video, a quote, a document or a number.

Answer NO in all other cases. For example:
- The claim is about the same subject, but it says something different.
- The claim is about a completely different subject.

Rules
1. Base your answer on the claim. The explanation, the organisation, its country and the date only help you understand who or what the claim talks about. For example, a claim about "this video" is about the video that the explanation describes. The background only helps you understand the query. Do not use any of them to add something that the claim does not say.
2. Do not think about whether the claim is true or false. The fact-checker's verdict never matters.
3. Judge each claim on its own. You may use what you generally know (for example who a famous person is). Do not search the web.
4. If the claim is empty, or you still cannot understand it after reading the start of the explanation, answer NO.

Examples (invented, on subjects that are not among the queries)
Query: "The Moon landings were filmed in a studio."
- "NASA faked the Apollo 11 landing on a film set." YES: the same thing in other words.
- "The Moon landings were filmed in a studio, and NASA hid the proof." YES: the claim contains what the query says, plus something else.
- "Were the Moon landings filmed in a studio?" YES: the claim asks whether it is true.
- "The Moon landings were not filmed in a studio." YES: the claim says that it is false.
- "A photo of the flag moving in the wind proves that the landing was staged." YES: the claim tries to prove it.
- "The Apollo astronauts were paid 8 dollars a day." NO: same subject, something different.
- "Venice will charge tourists an entry fee." NO: a completely different subject.
Query: "Tourists are destroying Venice."
- "A tourist carved his name into a bridge in Venice." YES: one specific case.
- "A man damaged a bridge in Venice." NO: the claim does not say that the man was a tourist.
Query: "Venice had 30 million tourists in 2019."
- "Venice had 31 million tourists in 2019." YES: the same statement. The number differs a little.
Query: "The 2019 fire at Notre-Dame cathedral was started on purpose."
- "Arsonists set Notre-Dame on fire in 2019." YES: the same fire, the same statement.
- "The 2020 fire at Nantes cathedral was started on purpose." NO: another fire.
Query: "The mayor of Springfield banned bicycles." Background: "The mayor of Springfield is Joe Quimby."
- "Quimby outlawed bikes in his city." YES: the background tells you that Quimby is the mayor.
- "Joe Quimby is corrupt." NO: same person, something different."""

_CASE = "Now judge this case."
_ANSWER = "Answer with one word: YES or NO."

# A claim longer than this is cut (and the label records it); real claims are far shorter. It keeps every prompt
# inside the 4,096-token context of model_selection/RULES.md, section 4.
CLAIM_MAX_CHARS = 3000


def _one_line(text: str | None) -> str:
    return " ".join((text or "").replace("\r", " ").split())


def judge_prompt(query: dict, item: dict) -> tuple[str, bool]:
    """The whole prompt for one (query, document) pair, and whether the claim had to be cut.
    The background line appears only when the query has a background, as in the human judges' sheet."""
    claim = _one_line(item.get("claim_text")) or "(empty)"
    cut = len(claim) > CLAIM_MAX_CHARS
    if cut:
        claim = claim[:CLAIM_MAX_CHARS] + "..."
    lines = [_GUIDE, "", _CASE, f"Query: {_one_line(query.get('free_text'))}"]
    background = _one_line(query.get("background"))
    if background:
        lines.append(f"Background: {background}")
    lines += [f"Claim: {claim}",
              f"Start of the explanation: {_one_line(item.get('article_start')) or '(none)'}",
              f"Organisation: {item.get('author_name') or '?'}, country: {item.get('country') or '?'}, "
              f"date: {item.get('date_published') or '?'}",
              "", _ANSWER]
    return "\n".join(lines), cut


def prompt_fingerprint() -> str:
    """sha256 of the fixed parts of the prompt: every label file records it, so a changed prompt is visible."""
    return hashlib.sha256((_GUIDE + "\n" + _CASE + "\n" + _ANSWER).encode("utf-8")).hexdigest()
