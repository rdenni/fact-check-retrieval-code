# The formula writers' instruction

English translation of the page the two formula writers received in Italian, with a sheet of one row per query. The Supplementary Material prints it from the first numbered heading on (reproduce_paper.py copies that part).

## 1. The task

There is an Excel sheet with a list of narratives: short descriptions, in plain words, of a disinformation story, for example "Western weapons sent to Ukraine are resold on the black market". For each narrative you have to write the search formula you would use to find all the fact-checks about that narrative in an archive of fact-checks, using Wikipedia pages (explained better below). Write the formula you would really use.

## 2. How this search works

Every fact-check article in the archive was processed by an automatic tool that marks the English Wikipedia pages mentioned in the article's text. For example an article about weapons smuggling from Ukraine may be linked to the pages Ukraine, Black market, Arms trafficking. The search takes a formula made of page titles combined with AND, OR and NOT and returns all the fact-checks whose linked pages satisfy the formula, newest first.

So a concept that has no Wikipedia page (a relation such as "X causes Y", a vague idea such as "corruption of the elites") cannot be searched. If you look for a page for a concept and do not find it, write the concept in the column not_expressible (briefly, if possible). It takes a few seconds and it is one of the things the study measures.

## 3. Which pages you can use: any page of English Wikipedia

Open https://en.wikipedia.org, find the page of the concept and copy the title shown at the top of the page you land on. Wikipedia often redirects: typing Bioweapon lands on the page Biological agent, so write Biological agent. Write the title exactly (capitals, accents, singular or plural as on the page): copy and paste, it is easier.

## 4. How the formula is written

Write every page title between double quotes and combine the titles with AND, OR, NOT and parentheses:

```
("Ukraine" OR "Armed Forces of Ukraine") AND ("Black market" OR "Arms trafficking") AND NOT "Video game"
```

- The quotes are needed because a title can contain the word and ("Trinidad and Tobago") or parentheses ("Mercury (planet)"). A title without quotes is a format error and the row comes back to you.
- AND narrows: both sides must be there. Use one AND group for every concept the narrative requires.
- OR widens: one of the two sides is enough. Use it inside a group for synonyms, related pages and specific cases ("Ukraine" OR "Armed Forces of Ukraine").
- NOT removes fact-checks from a positive set (A AND NOT B). Never write a formula that is only a NOT, or an OR with a NOT inside: "everything except" is not a search and the system refuses it.
- Use parentheses when you use AND.
- A formula can have as many pages as you want; a single page is also a valid formula.

### Examples: what a formula returns

Imagine four fact-checks of the archive (invented; in the real one every article has tens of linked pages) and the pages the tool linked to each:

| fact-check | linked pages |
|---|---|
| FC1 | Ukraine, Arms trafficking, Black market |
| FC2 | Ukraine, NATO |
| FC3 | Mexico, Arms trafficking |
| FC4 | Ukraine, Arms trafficking, Video game |

Here is what some formulas return on these four:

| formula | result | why |
|---|---|---|
| "Ukraine" | FC1, FC2, FC4 | all the fact-checks linked to the page Ukraine |
| "Ukraine" AND "Arms trafficking" | FC1, FC4 | AND: both pages are needed |
| "Ukraine" OR "Arms trafficking" | FC1, FC2, FC3, FC4 | OR: one of the two is enough |
| "Ukraine" AND "Arms trafficking" AND NOT "Video game" | FC1 | as the second, minus those linked to Video game |
| ("Ukraine" OR "Mexico") AND "Arms trafficking" | FC1, FC3, FC4 | either country, plus the arms-trafficking page |
| "Ukraine" OR "Mexico" AND "Arms trafficking" (no parentheses) | FC1, FC2, FC3, FC4 | AND binds tighter than OR, so the system reads it as "Ukraine" OR ("Mexico" AND "Arms trafficking") and FC2 comes in. Parentheses avoid the ambiguity: always use them when you mix AND and OR |
| "Ukraine" AND "Arms trafficking" AND "Black market" AND "NATO" | none | too many ANDs: no fact-check has all four pages |
| NOT "Video game" | refused by the system | "everything except" is not a search |

Rule of thumb: one AND group for every concept that must be there; inside every group, OR between the pages that express the same concept (synonyms, specific names, neighbouring pages); NOT only to remove a wrong meaning that you know would pollute the results. Every extra AND loses fact-checks, every extra OR lets more in.

A formula usually takes five to fifteen minutes. Refine it as much as you want before submitting it; after that it is final.

## 5. The sheet: one row per narrative

The sheet comes with the first three columns already filled: query_id (the narrative's number, do not change it), author_id (your code) and narrative (the text, do not edit it; if it is unclear say so in notes). You fill the other five:

| column | what to write |
|---|---|
| formula | the formula, titles in double quotes |
| not_expressible | the concepts you looked for a page for without finding one, separated by semicolons (empty if none) |
| minutes | the time from reading the narrative to the final formula, in minutes (start a timer) |
| refinements | how many times you changed the formula after the first complete version (0 if never) |
| notes | optional: for example "narrative unclear", or anything that bothered you |

If a narrative is unclear or you think it is wrong, write it in notes and still write your best formula.

## 6. Rules of independence

- Do not discuss the narratives or your formulas with the other person who writes the formulas until you are told that you may.
- Do not look for the fact-checks or the formula in the archive, on fact-checking sites, on search engines or with an AI assistant. Wikipedia is your only tool.
- Do not change a formula after submitting it, even if a later narrative makes you think of a better one.
