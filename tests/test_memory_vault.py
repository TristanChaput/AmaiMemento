import tempfile
import unittest
from pathlib import Path

from amai_memory.vault import iter_vault_documents


class VaultDiscoveryTests(unittest.TestCase):
    def test_discovers_active_markdown_and_reads_minimal_frontmatter(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            note = vault / "10-Projects" / "AmaiAgent" / "Context.md"
            note.parent.mkdir(parents=True)
            note.write_text(
                "---\ntype: context\nprojet: AmaiAgent\n---\n# Context\n",
                encoding="utf-8",
            )
            documents = list(iter_vault_documents(vault))
            self.assertEqual(
                [document.relative_path for document in documents],
                ["10-Projects/AmaiAgent/Context.md"],
            )
            self.assertEqual(documents[0].note_type, "context")
            self.assertEqual(documents[0].project, "AmaiAgent")
            self.assertEqual(len(documents[0].content_hash), 64)

    def test_excludes_archives_and_obsidian_internal_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            for relative in ["90-Archives/old.md", ".obsidian/plugins/cache.md"]:
                path = vault / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# Ignore me", encoding="utf-8")
            active = vault / "00-Inbox" / "Inbox.md"
            active.parent.mkdir(parents=True)
            active.write_text("# Keep me", encoding="utf-8")
            documents = list(iter_vault_documents(vault))
            self.assertEqual(
                [document.relative_path for document in documents],
                ["00-Inbox/Inbox.md"],
            )


if __name__ == "__main__":
    unittest.main()
