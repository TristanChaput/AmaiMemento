# AmaiMemory

CLI and library for the local **Memora** semantic index built from the
AmaiKnowledge vault. This repository was extracted from AmaiAgent so it can be
used independently by the other Amai projects.

## Context: AmaiKnowledge

AmaiKnowledge is a private [Obsidian](https://obsidian.md) vault: it is the core
of my memory, covering all of my projects as well as my personal life. It is not
published; this repository only contains the tooling that indexes it, never its
content.

AmaiMemory reads the vault's Markdown notes (`../AmaiKnowledge` by default,
configurable through `AMAI_KNOWLEDGE_PATH`) and builds a local semantic index so
that Amai agents and tools can query this memory. The notes remain the canonical
source; `data/memora/` only holds a derived, rebuildable index that is excluded
from the repository.

To use it with your own Obsidian vault, point `AMAI_KNOWLEDGE_PATH` at its
folder.

## Installation

Memora and the embedding models have heavy dependencies. Use Python 3.12:

```powershell
git submodule update --init vendor/Memora
uv venv --python 3.12
uv pip install -e .
uv pip install -r vendor/Memora/requirements.txt
```

Copy `.env.example` to `.env` and set `OPENCODE_API_KEY`.

## Usage

```powershell
uv run amai-memory scan
uv run amai-memory status
uv run amai-memory rebuild --workers 2
uv run amai-memory query "What are the long-standing constraints of AmaiTalk?" --top-k 5
uv run amai-memory benchmark-embeddings "What are the long-standing constraints of AmaiTalk?"
```

See [`docs/memora.md`](docs/memora.md) for installation, rebuilding and
searching.

## License

[MIT](LICENSE). Memora (`vendor/Memora`) is distributed by Microsoft under its
own MIT license.
