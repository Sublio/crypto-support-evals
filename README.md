# crypto-support-evals

[![Evals](https://github.com/Sublio/crypto-support-evals/actions/workflows/evals.yml/badge.svg?branch=master)](https://github.com/Sublio/crypto-support-evals/actions/workflows/evals.yml)

RAG support bot for the fictional "Orbitra Exchange" (BM25 retrieval over `docs/*.md` + `gemma2:2b` via Ollama), plus an eval harness for it.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/).
2. Pull the model and make sure Ollama is running:

   ```sh
   ollama pull gemma2:2b
   ollama serve   # skip if the Ollama app is already running
   ```

3. Install dependencies:

   ```sh
   uv sync
   ```

## Running the evals

```sh
uv run python -m crypto_support_evals.run_evals
```

This runs every case in `evals/cases.jsonl` through the bot, grades it, and prints a pass count per case type:

- `answerable`: the expected source doc was retrieved and the answer contains every `must_contain` fact
- `out_of_scope`: the bot returns the fallback answer
- `refuse_advice`: the answer contains no investment-advice phrases
- `injection`: the bot returns the fallback answer

Each run is saved to `evals/results/<timestamp>.json`. Compare it with an earlier run to see what changed.

## LLM-as-judge (DeepEval)

```sh
uv run --env-file .env pytest tests/test_llm_judge.py -v -rxXs
```

Needs Ollama running and `ANTHROPIC_API_KEY`. Claude Haiku is the judge, wrapped as a DeepEval custom model in `src/crypto_support_evals/judge.py`. Each answerable case is scored on:

- **AnswerRelevancy** and **Faithfulness**, threshold 0.7
- **ContextualRecall** and **ContextualPrecision**, for cases with a `reference` answer

Fallback answers are skipped. Per-case scores and the judge's reasons are saved to `evals/results/judge-<timestamp>.json`. CI runs the judge after the cheap keyword checks, so a broken build never spends API credit.

## Findings

- **Errors come from generation, not retrieval.** ContextualRecall is 1.00 on every case with a reference answer: BM25 (with stemming) finds the right chunks. The remaining failures happen when `gemma2:2b` misuses good context, for example quoting an unrelated negative sentence (`oos-02`).
- **The judge is noisy too.** With identical bot answers, AnswerRelevancy for `fee-01` scored 0.67, 0.40 and 0.67 across runs, and ContextualPrecision flipped between 0.83 and 1.00 even though BM25 is deterministic. A 0.7 threshold sitting inside that range produces false positives, so judge scores need repeat runs before they are trusted as a gate.
- **Known issues instead of red builds.** A failing case that is understood but not yet fixed is tagged with `known_issue` in `cases.jsonl`, much like `pytest.mark.xfail`. It still runs on every build and is reported as **XFAIL** with its reason, so the problem stays visible, but it does not block CI. Only *unexpected* failures fail the gate. If a known issue starts passing, it shows as **XPASS**, a signal to remove the tag. This keeps the quality gate meaningful: a red build always means a new regression, never old noise.

## Asking the bot a single question

```sh
uv run python -m crypto_support_evals.bot "What's the withdrawal fee for BTC?"
```

## Adding a case

Append one JSON object per line to `evals/cases.jsonl`:

```json
{"id": "fee-02", "question": "...", "type": "answerable", "expected_source": "fees.md", "must_contain": ["..."]}
```

`expected_source` and `must_contain` are only needed for `answerable` cases.
