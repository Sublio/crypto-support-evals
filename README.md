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
