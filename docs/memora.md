# Memora for AmaiKnowledge

Memora is a local, optional layer. The AmaiKnowledge Markdown notes remain the canonical source; `data/memora/` only contains a derived, rebuildable index.

The Microsoft code is pinned as a Git submodule. The integration indexes every active Markdown note, excluding `90-Archives`, `.obsidian`, hidden folders and non-Markdown files. The path, content hash, note type and project are passed along as provenance.

## Installation

Memora and the embedding models have heavy dependencies. Use Python 3.12:

```powershell
git submodule update --init vendor/Memora
uv venv --python 3.12
uv pip install -e .
uv pip install -r vendor/Memora/requirements.txt
```

To use `google/embeddinggemma-2`, add `torchvision`: the model's multimodal processor is loaded even in text-only mode and depends on it.

```powershell
uv pip install torchvision
```

The first `rebuild` or `query` may download the `google/embeddinggemma-2` weights locally. The model runs on CPU and produces normalized 768-dimensional vectors.

Create or complete `.env` without ever committing the key:

```dotenv
OPENCODE_API_KEY=
AMAI_KNOWLEDGE_PATH=../AmaiKnowledge
MEMORA_DATA_PATH=./data/memora
MEMORA_LLM_MODEL=glm-5.2
MEMORA_EMBEDDING_MODEL=google/embeddinggemma-2
MEMORA_REBUILD_WORKERS=2
MEMORA_ENABLE_CUE_INDEX=false
MEMORA_EMBEDDING_CACHE_SIZE=8192
```

## Faster rebuilds

The remote extraction phase can run across several workers, while
deduplication and Chroma writes stay sequential to avoid races between updates.
Identical embeddings are cached for the lifetime of the process, and provenance
is recorded without rescanning the whole collection after each file.

```powershell
uv run amai-memory rebuild --workers 2
```

Each completed note is saved in a checkpoint. After an interruption, resume
after the last note printed, using its relative path:

```powershell
uv run amai-memory rebuild --resume-after "10-Projects/Example/Some note.md"
```

Resuming checks that the vault content and the cue index configuration have not
changed. Older builds interrupted without a checkpoint must be rebuilt from
scratch.

`MEMORA_REBUILD_WORKERS` sets the default value. Start at `2`, then measure
before increasing it. Cue indices are now disabled by default: they accounted
for most of the vectors and also require extra LLM generation. Use `--cues` (or
`MEMORA_ENABLE_CUE_INDEX=true`) to favor search richness over speed.
`MEMORA_EMBEDDING_CACHE_SIZE` controls how many embeddings are kept in memory
during the process (`8192` by default, `0` disables the cache). The manifest
records the configuration, the duration and the number of workers used.

`OPENCODE_API_KEY` is only required for `rebuild`. Extraction of abstractions and cue anchors uses `MEMORA_LLM_MODEL` through OpenCode Go (`https://opencode.ai/zen/go/v1`). GPT models, including `gpt-6-luna`, use `/responses`; GLM uses `/chat/completions`. Embeddings stay local with Sentence Transformers.

## Usage

```powershell
# Preview without a key and without loading the embedding model
uv run amai-memory scan

# Fresh rebuild, activated only on success
uv run amai-memory rebuild

uv run amai-memory status
uv run amai-memory query "What are the long-standing constraints of AmaiTalk?" --top-k 5
```

The long-lived HTTP server was deliberately left out of this repository when it
was decoupled from AmaiAgent. The canonical interface is the CLI command, which
reloads the process on every call.

The observed profile is still about 32 s for the first load, then 4–5 s per
embedding/search on CPU. `HF_TOKEN` can speed up the initial download but does
not reduce this CPU cost.

`query` makes no remote call and does not require `OPENCODE_API_KEY` once the index is built. With the default model, it computes the question embedding locally using Gemma's `SearchQuery` prompt; memories use the `Document` prompt.

## Choosing the embedding model

Two local models are supported through `MEMORA_EMBEDDING_MODEL`:

- `google/embeddinggemma-2` (default): 768 dimensions, text + code. Loaded in
  text-only mode (270M) via `config_kwargs={"vision_config": None, "audio_config": None}`.
  Queries use the `SearchQuery` prompt, memories the `Document` prompt.
- `Qwen/Qwen3-Embedding-0.6B`: 1024 dimensions, multilingual.

The model in use is recorded in `current.json` (`configuration.embedding_model`)
and in the checkpoint. An index built with one model cannot be queried with the
other: the vectors are not comparable. Switching models therefore requires a
full `rebuild`, and a checkpoint resume cannot change the model.

## Comparing two embedding models

`benchmark-embeddings` reuses the memories already extracted in the active index
(no `rebuild`, no LLM call, no loading of the index's model) and compares, for
each model, latency, memory and search ranking on probe questions. Each model
runs in its own process, which isolates its resident memory (`model_rss_mb`,
`total_rss_mb`) from the others. Without a question, a sample of the indexed
memories is used as probes.

```powershell
uv run amai-memory benchmark-embeddings
uv run amai-memory benchmark-embeddings "What are the constraints of AmaiTalk?" "How does Memora work?" --top-k 5
```

The JSON output contains, per model, document and query computation times,
resident memory and the top-k results. A `summary` section gives the mean
overlap at k and the top-1 agreement rate relative to the first model
(`reference`). These metrics measure divergence between models, not absolute
quality: validate the results manually before migrating.

## Safety

Each rebuild writes into a new folder and only updates `current.json` after it succeeds. An interrupted rebuild therefore never disables the previous index. The OpenCode Go quota is consumed during `rebuild`, not during `query`.

Search is not yet published in the Hermes tool registry. That exposure should be decided after evaluating result quality and cost.
