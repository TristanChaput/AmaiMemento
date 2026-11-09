---
name: memora-query
description: Query the local Memora semantic index built from the AmaiKnowledge vault. Use it to retrieve cross-cutting context about projects, decisions, constraints, learnings or notes from the vault, notably from AmaiAgent or AmaiTalkWebApp, and whenever the user asks to use Memora or the local memory.
---

# Query Memora

Resolve the AmaiMemento repository from any sibling repository:

```powershell
$amaiMemory = Join-Path (Split-Path (git rev-parse --show-toplevel)) "AmaiMemento"
$env:PYTHONUTF8 = "1"
```

## Search

1. Check that an index is active:

   ```powershell
   uv run --directory $amaiMemory amai-memory status
   ```

2. Check `configuration.embedding_model` in the `status` JSON. If the status is `not_built` or the model differs from `google/embeddinggemma-2`, search AmaiKnowledge directly and report that the index must be rebuilt. Never run `rebuild` implicitly.

3. Write a precise question and query the index:

   ```powershell
   uv run --directory $amaiMemory amai-memory query "<question>" --top-k 5
   ```

4. Read the notes listed in `sources` before any change or claim that requires the canonical text. Memora helps find information; the vault's Markdown remains the canonical source.

## Other commands

- `amai-memory scan`: preview the indexable notes, without a key or model.
- `amai-memory rebuild`: rebuild the index only on explicit request; this consumes the OpenCode Go quota and requires `OPENCODE_API_KEY`.
- `amai-memory rebuild --resume-after "<relative path>"`: resume an interrupted build after the last completed note printed, if its checkpoint exists and the vault has not changed.

`query` stays local once the index is built, but the first use may load or download `google/embeddinggemma-2`, the default model. The active index records its own model; switching to Gemma requires a rebuild.
