from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml


EXCLUDED_DIRECTORIES = {".git", ".obsidian", ".smart-env", "90-Archives"}


@dataclass(frozen=True)
class VaultDocument:
    path: Path
    relative_path: str
    content: str
    content_hash: str
    note_type: str
    project: str

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "source_path": self.relative_path,
            "source_hash": self.content_hash,
            "note_type": self.note_type,
            "project": self.project,
        }


def _frontmatter(content: str) -> dict:
    if not content.startswith("---"):
        return {}
    lines = content.splitlines()
    try:
        closing_line = lines.index("---", 1)
    except ValueError:
        return {}
    parsed = yaml.safe_load("\n".join(lines[1:closing_line])) or {}
    return parsed if isinstance(parsed, dict) else {}


def iter_vault_documents(vault_path: Path) -> Iterator[VaultDocument]:
    root = vault_path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AmaiKnowledge vault not found: {root}")

    for path in sorted(root.rglob("*.md")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRECTORIES or part.startswith(".") for part in relative.parts[:-1]):
            continue
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue
        properties = _frontmatter(content)
        yield VaultDocument(
            path=path,
            relative_path=relative.as_posix(),
            content=content,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            note_type=str(properties.get("type", "")),
            # The vault frontmatter uses the French key "projet".
            project=str(properties.get("projet", "")),
        )

