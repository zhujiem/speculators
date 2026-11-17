# PR: CustomDataset — support OpenAI `messages` rows with tools & per-row sampling params

**Repo:** `/vllm-workspace/vllm` (branch working tree)
**File changed:** `vllm/benchmarks/datasets/datasets.py` (+112 / −13)
**Patch:** `0003-custom-dataset-openai-messages-support.patch`

## Summary

Extends the benchmark `CustomDataset` (JSONL) to accept OpenAI Chat Completions–shaped rows, in addition to the existing plain `prompt` column. This lets `benchmark_serving` replay production-style traffic that contains chat messages, tool definitions, tool calls, and per-request sampling parameters.

## Changes

### 1. JSONL loader: pandas → `json` module

`pd.read_json(lines=True)` normalizes nested lists of dicts (e.g. `messages`, `tools`) into per-column dicts, corrupting OpenAI-shaped rows. The loader now parses each line with `json.loads`, preserving the row shape, and collects the union of column names.

Validation now accepts a `prompt` **or** `messages` column (previously only `prompt`), with an error message describing both formats and the optional `tools` / `tool_choice` columns.

### 2. New module-level helper `_parse_tool_call()`

Copies a tool call and parses its `function.arguments` from the JSON-string wire form to a dict — mirroring what the vLLM server does before applying a chat template. Tolerates malformed JSON (leaves the string as-is) and never mutates the original row data.

### 3. `sample()`: per-row dispatch

- **Prompt resolution:** each row must contain a `prompt` string or a `messages` list; otherwise a `ValueError` names the offending row index.
- **Tools:** when a row has `tools`, request overrides carry `tools` plus `tool_choice` (valid values: `"auto"`, `"required"`, or a named-tool dict; anything else defaults to `"auto"`).
- **Per-row sampling params:** `temperature`, `top_p`, `reasoning_effort` are forwarded as request overrides (per-request wins over CLI flags / `--extra-body` at dispatch time).
- **Length estimation for `messages` rows:** renders the messages with `apply_chat_template` (tools included, `reasoning_effort` passed so GLM-style templates match the server render) for an accurate `prompt_len`. `tool_calls` arguments are parsed via `_parse_tool_call` on copies, so the original messages stay untouched. Falls back gracefully:
  - `TypeError` → re-render without the `tools` kwarg (older templates);
  - other exceptions → log a warning and join message contents as plain text.
- **No tokenizer:** `prompt_len = 1` and an empty prompt, as before.
- **Request construction:** `SampleRequest` now receives `chat_messages=messages` and `request_overrides=overrides`, so the backend can issue Chat Completions requests with tools rather than raw prompts.

## Behavior notes

- Plain-`prompt` datasets behave exactly as before (template application, `skip_chat_template`, oversampling).
- Row validation error messages include the row index for quick debugging.

## Testing

- `python -m py_compile` passes.
- Unit sanity check of `_parse_tool_call` (direct import): parses JSON-string arguments, tolerates malformed JSON, returns copies (original untouched).
- Repo pytest could not run locally (missing `tblib` in the environment); recommend running in CI:
  - `tests/benchmarks/test_custom_dataset_seed.py`
  - `tests/benchmarks/test_custom_dataset_chat_template_kwargs.py`

## Applying the patch

```bash
cd /path/to/vllm
git apply 0003-custom-dataset-openai-messages-support.patch
```
