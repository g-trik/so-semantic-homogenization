# The Contraction of the Commons

Semantic homogenization on Stack Overflow after LLM emergence.

**Author:** G. Trikkas Britt
**Programme:** MSc Business Information Management, Rotterdam School of Management, Erasmus University Rotterdam, 2026

## Research question

Has the emergence of large language models led to increased semantic
homogenization of answers in online knowledge platforms?

**H1** Following ChatGPT's release, answers in high-resource languages become
more semantically homogeneous than answers in low-resource languages.

**H2** The increase is stronger in high-volume questions than in low-volume
questions.

**H3** The increase is stronger among highly visible (accepted) answers.

**H4** Contributions by less experienced (lower-reputation) users exhibit
stronger homogenization than those by more experienced users.

## Design

Difference-in-differences with two-way fixed effects, treating ChatGPT's
November 2022 release as an exogenous shock. Treatment tags are seven
programming languages where LLMs generate competent code (Python, JavaScript,
Java, C++, TypeScript, PHP, Ruby); control tags are two low-resource languages
(Bash, R), selected on MultiPL-E popularity tiers and Codex pass@1 scores
(Cassano et al., 2023) and corroborated by Joel, Wu and Fard (2025).

The analysis runs at two scales and across three content types, giving six
analytical cells:

|  | prose only | prose & code | code only |
|---|---|---|---|
| **tag level** | 756 (tag, month) clusters | 756 | 756 |
| **question level** | 8,822 questions | 8,822 | 8,822 |

Semantic homogenization is the mean pairwise cosine similarity of answer
embeddings within a cluster, computed with OpenAI's text-embedding-3-large
(3,072 dimensions). A single embedding model is used for all content types, so
prose/code contrasts are identified on content rather than on the model. The
analysis is restricted to answers containing both prose and code, so the three
content types are three extractions from the same 74,742 (tag level) and
66,650 (question level) answers.

## Finding

No statistically significant treatment effect in any of the six cells. Across
21 hypothesis tests, no result survives multiple-comparison correction under
the reduced specification; one survives under the saturated specification but
is not separately identified from its constituent two-way term. The three
moderator hypotheses are reported as not identified rather than unsupported,
owing to a treatment-to-control ratio of 20.3:1 at question level.

## Pipeline

| Script | Purpose |
|---|---|
| `01_parse_posts.py` | parses Posts.xml into three answer corpora |
| `02_parse_users.py` | parses Users.xml (reputation, account creation) |
| `03_parse_votes.py` | parses Votes.xml |
| `04_parse_tags.py` | parses Tags.xml |
| `05_join.py` | joins answers with accepted flags, reputation, tags |
| `06_common_ids.py` | builds the common subsample |
| `07_sample_v3.py` | tag classification and panel construction |
| `08_embed_v3.py` | embedding generation |
| `09_metrics_v3.py` | cluster-level similarity and covariates |
| `10_master_v3.py` | primary regressions, all four hypotheses |

`08_embed_v3.py` requires `OPENAI_API_KEY` in the environment.

## Data

Stack Exchange Data Dump 2025-12-31
https://archive.org/details/stackexchange_20251231

Intermediate parquet files and embeddings are not included; the pipeline
regenerates them from the dump.

## Licence

Code: MIT. Results and figures: CC BY 4.0.

Cite as: Trikkas Britt, G. (2026). *The contraction of the commons: Has the
emergence of large language models led to increased semantic homogenization of
answers in online knowledge platforms?* MSc thesis, Rotterdam School of
Management, Erasmus University Rotterdam.

Raw Stack Overflow data is licensed CC BY-SA 4.0 by Stack Exchange Inc.
