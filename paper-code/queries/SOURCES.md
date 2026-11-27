# Where the passages of reports quoted in this folder come from

The reports of the observers are not part of this repository (`queries/fetch_pages.py` and `queries/fetch_edmo_briefs.py`
fetch them; `queries/report_sources.csv` lists them with their addresses). A few files quote short passages of them, word
for word, as examples for a research use, and name the source here: the prompts are released exactly as the paper's runs
used them (the checksum of the extraction prompt is recorded with every query), and the review rules as they were applied,
so the sources are not written into them.

## `prompts/rewrite_needs.txt`

Four passages of the monthly fact-checking briefs of the European Digital Media Observatory (EDMO, edmo.eu; © EDMO, all
rights reserved).

| passage (its first words) | source |
|---|---|
| Ice at the South Pole grew by 17% since 1979 | EDMO monthly fact-checking brief no. 44, 17 February 2025, https://edmo.eu/publications/monthly-fact-checking-brief-no-44-is-out/ |
| unfounded information spread the message that the EU had approved the sale of lab-grown meat | EDMO monthly fact-checking brief no. 33, 15 March 2024, https://edmo.eu/publications/monthly-fact-checking-brief-no-33-is-out/ |
| July also saw the comeback of a widely spread transmisogynistic conspiracy theory | EDMO monthly fact-checking brief no. 50, 22 August 2025, https://edmo.eu/publications/monthly-fact-checking-brief-no-50-is-out/ |
| A link circulated to a website, allegedly from the French Ministry of the Armed Forces | EDMO monthly fact-checking brief no. 34, 17 April 2024, https://edmo.eu/publications/monthly-fact-checking-brief-no-34-is-out/ |

## `prompts/draft_needs.txt`

The extraction prompt (printed in the Supplementary Material) quotes no report word for word. One of its examples is
adapted from Freedom House, *Freedom on the Net 2024: India* (https://freedomhouse.org/country/india/freedom-net/2024),
which reports a Washington Post investigation of a network of 150,000 workers; in the example the party has no name.

## `REVIEW-RULES.md`

The rules show how they were applied with claims and short passages of the reports (each at most 30 words), in the
order of the file:

| quotation (its first words) | report | source |
|---|---|---|
| A BJP MLA called for a boycott of Muslim vegetable vendors | freiheit-india-infodemic-2020 | https://www.freiheit.org/india/misinformation-and-healthcare-infodemic-india |
| The ruling party was allegedly using an app called Tek Fog | fotn-india-2022 | https://freedomhouse.org/country/india/freedom-net/2022 |
| AI-generated image of Keir Starmer shared to suggest Labour | cetas-uk-elections-2024 | https://cetas.turing.ac.uk/publications/ai-enabled-influence-operations-threat-analysis-2024-uk-and-european-elections |
| A study shows that human CO2 emissions don't cause current global warming | report-02 | EDMO monthly report no. 2, 2026-08-25, https://edmo.eu/publications/monthly-report-no-2-is-out/ |
| A Dominion Voting Systems whistleblower accused US politicians | newsguard-2026-elections-tracker | https://www.newsguardtech.com/special-reports/2026-elections-misinformation-tracker/ |
| S. Kasselakis and T. Macbeth naked on a beach | cetas-uk-elections-2024 | https://cetas.turing.ac.uk/publications/ai-enabled-influence-operations-threat-analysis-2024-uk-and-european-elections |
| Moderna's new potential Covid-19 vaccine will change people's DNA | firstdraft-under-the-surface-2020 | https://firstdraftnews.org/wp-content/uploads/2020/11/FirstDraft_Underthesurface_Fullreport_Final.pdf |
| Karachi port alleges this was the result of an account-level hack | isd-india-pakistan-2025 | https://www.isdglobal.org/digital-dispatch/missiles-and-misinformation-false-claims-about-the-india-pakistan-clashes-reach-millions-on-x/ |
| The government denied the occurrence of shootings during the #EndSARS Protests | fotn-nigeria-2022 | https://freedomhouse.org/country/nigeria/freedom-net/2022 |
| his position as chair of the House Foreign Affairs Committee | mtac-election-report-5-2024 | https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/msc/documents/presentations/CSR/MTAC-Election-Report-5-on-Russian-Influence.pdf |
| he was falsely said to have financed the Central American caravan | shorenstein-midterms-2018 | https://shorensteincenter.org/article/midterms-2018-trends-information-disorder/ |
| The proposed legislation would not provide the ACMA with content removal powers | fotn-australia-2024 | https://freedomhouse.org/country/australia/freedom-net/2024 |
| A cancer hospital would not accept new Muslim patients | freiheit-india-infodemic-2020 | https://www.freiheit.org/india/misinformation-and-healthcare-infodemic-india |
| A cancer hospital in the country also put out a notice | freiheit-india-infodemic-2020 | https://www.freiheit.org/india/misinformation-and-healthcare-infodemic-india |
| President Barack Obama himself was illegal, and therefore treasonous | senate-ira-new-knowledge-2018 | https://www.intelligence.senate.gov/sites/default/files/documents/The-Tactics-and-Tropes-of-the-Internet-Research-Agency.pdf |
| Rashida Tlaib is guilty by association with Islamic organizations | shorenstein-midterms-2018 | https://shorensteincenter.org/article/midterms-2018-trends-information-disorder/ |
| Hank Aaron’s tragic death is part of a wave of suspicious deaths | virality-project-2021 | https://stacks.stanford.edu/file/druid:mx395xj8490/Virality_project_final_report.pdf |
| The baseball legend received the Moderna vaccine on Jan. 5 | virality-project-2021 | https://stacks.stanford.edu/file/druid:mx395xj8490/Virality_project_final_report.pdf |
| Corbyn did not condemn bombings by the IRA | firstdraft-uk-election-2019 | https://firstdraftnews.org/articles/misinfo-types-uk-election/ |
| The official account of the Nazi Holocaust is a lie | hopenothate-state-of-hate-2023 | https://hopenothate.org.uk/state-of-hate |
| Hillary Clinton is a traitor! Hillary Clinton is a liar! | senate-ira-new-knowledge-2018 | https://www.intelligence.senate.gov/sites/default/files/documents/The-Tactics-and-Tropes-of-the-Internet-Research-Agency.pdf |

The quotations of papers in `REVIEW-RULES.md` and `model_selection/RULES.md` carry their reference where they stand.
