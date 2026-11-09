import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from amai_memory.memora_adapter import (
    EMBEDDINGGEMMA2_MODEL,
    MemoraUnavailableError,
    _load_memora,
    format_query,
)
from amai_memory.service import (
    MemoryPaths,
    build_index,
    current_manifest,
    query_index,
)
from amai_memory.vault import VaultDocument


class FakeMemoraClient:
    def __init__(self, query_results=None):
        self.memories = []
        self.added_content = []
        self.queries = []
        self.query_results = query_results or []
        self.list_calls = 0

    def add(self, content, type, metadata):
        self.added_content.append(content)
        entry = SimpleNamespace(index=f"Index {len(self.memories) + 1}", value=content)
        self.memories.append(entry)
        return [entry]

    def list_memories(self, limit):
        self.list_calls += 1
        return self.memories[:limit]

    def query(self, question, top_k, **kwargs):
        self.queries.append((question, top_k, kwargs))
        return self.query_results[:top_k]


class MemoryBuildTests(unittest.TestCase):
    def test_parallel_provenance_tracks_the_final_merged_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = MemoryPaths(vault=root / "vault", data=root / "data")
            document = VaultDocument(
                path=paths.vault / "note.md",
                relative_path="note.md",
                content="# Note",
                content_hash="abc",
                note_type="",
                project="",
            )
            second_document = VaultDocument(
                path=paths.vault / "other.md",
                relative_path="other.md",
                content="# Other",
                content_hash="def",
                note_type="",
                project="",
            )
            fake_client = FakeMemoraClient()

            class FakeAgentMemory:
                def add(self, entry):
                    fake_client.memories.append(entry)

            agent_memory = FakeAgentMemory()

            class FakeStorageBuilder:
                def upsert_memory_entry(self, entry):
                    agent_memory.add(
                        SimpleNamespace(index="merged-index", value=entry.value)
                    )

            cfg = SimpleNamespace(
                memory={
                    "enable_episodic_memory": False,
                    "enable_cue_index": False,
                }
            )
            fake_client._client = SimpleNamespace(
                cfg=cfg,
                _agent_memory=agent_memory,
                _model_client=object(),
                memory_builder_registry=SimpleNamespace(
                    get=lambda *args: FakeStorageBuilder()
                ),
            )

            class FakeExtractor:
                def __init__(self, *args, **kwargs):
                    pass

                def __call__(self, source_document):
                    return [
                        SimpleNamespace(
                            index="extracted-index",
                            value=source_document.content,
                        )
                    ]

            with (
                patch(
                    "amai_memory.service.create_client",
                    return_value=fake_client,
                ),
                patch(
                    "amai_memory.service._ParallelDocumentExtractor",
                    FakeExtractor,
                ),
            ):
                manifest = build_index(
                    paths,
                    [document, second_document],
                    workers=2,
                )

            provenance = json.loads(
                Path(manifest["provenance_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                provenance["merged-index"][0]["source_path"],
                "note.md",
            )
            self.assertEqual(fake_client.list_calls, 0)

    def test_parallel_rebuild_extracts_before_sequential_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = MemoryPaths(vault=root / "vault", data=root / "data")
            documents = [
                VaultDocument(
                    path=paths.vault / f"note-{index}.md",
                    relative_path=f"note-{index}.md",
                    content=f"# Note {index}",
                    content_hash=str(index),
                    note_type="",
                    project="",
                )
                for index in (1, 2)
            ]
            fake_client = FakeMemoraClient()
            persisted = []

            class FakeStorageBuilder:
                def upsert_memory_entry(self, entry):
                    persisted.append(entry.index)
                    fake_client.memories.append(entry)

            storage_builder = FakeStorageBuilder()
            cfg = SimpleNamespace(
                memory={
                    "enable_episodic_memory": False,
                    "enable_cue_index": False,
                }
            )
            local_client = SimpleNamespace(
                cfg=cfg,
                _agent_memory=object(),
                _model_client=object(),
                memory_builder_registry=SimpleNamespace(
                    get=lambda *args: storage_builder
                ),
            )
            fake_client._client = local_client

            class FakeExtractor:
                def __init__(self, *args, **kwargs):
                    pass

                def __call__(self, document):
                    return [
                        SimpleNamespace(
                            index=document.relative_path,
                            value=document.content,
                        )
                    ]

            with (
                patch(
                    "amai_memory.service.create_client",
                    return_value=fake_client,
                ),
                patch(
                    "amai_memory.service._ParallelDocumentExtractor",
                    FakeExtractor,
                ),
            ):
                manifest = build_index(paths, documents, workers=2)

            self.assertEqual(persisted, ["note-1.md", "note-2.md"])
            self.assertEqual(fake_client.added_content, [])
            self.assertTrue(manifest["rebuild"]["parallel_extraction"])
            self.assertEqual(manifest["rebuild"]["workers"], 2)
            self.assertEqual(fake_client.list_calls, 0)
            self.assertFalse(manifest["configuration"]["cue_index"])

    def test_build_activates_manifest_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = MemoryPaths(vault=root / "vault", data=root / "data")
            document = VaultDocument(
                path=paths.vault / "note.md",
                relative_path="note.md",
                content="# Note",
                content_hash="abc",
                note_type="ressource",
                project="",
            )
            with patch(
                "amai_memory.service.create_client", return_value=FakeMemoraClient()
            ):
                manifest = build_index(paths, [document])

            self.assertEqual(current_manifest(paths), manifest)
            provenance = json.loads(
                Path(manifest["provenance_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["Index 1"][0]["source_path"], "note.md")

    def test_failed_build_does_not_replace_active_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = MemoryPaths(vault=root / "vault", data=root / "data")
            paths.data.mkdir(parents=True)
            paths.manifest.write_text('{"build_id": "previous"}', encoding="utf-8")
            document = VaultDocument(
                path=paths.vault / "note.md",
                relative_path="note.md",
                content="# Note",
                content_hash="abc",
                note_type="",
                project="",
            )
            failing_client = FakeMemoraClient()
            failing_client.add = lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("boom")
            )

            with patch(
                "amai_memory.service.create_client", return_value=failing_client
            ):
                with self.assertRaises(RuntimeError):
                    build_index(paths, [document])

            self.assertEqual(current_manifest(paths), {"build_id": "previous"})

    def test_missing_key_does_not_replace_active_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = MemoryPaths(vault=root / "vault", data=root / "data")
            paths.data.mkdir(parents=True)
            paths.manifest.write_text('{"build_id": "previous"}', encoding="utf-8")
            document = VaultDocument(
                path=paths.vault / "note.md",
                relative_path="note.md",
                content="# Note",
                content_hash="abc",
                note_type="",
                project="",
            )

            with patch.dict(os.environ, {"OPENCODE_API_KEY": ""}):
                with self.assertRaises(MemoraUnavailableError):
                    build_index(paths, [document])

            self.assertEqual(current_manifest(paths), {"build_id": "previous"})

    def test_queries_use_gemma_prompt_but_documents_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = MemoryPaths(vault=root / "vault", data=root / "data")
            document = VaultDocument(
                path=paths.vault / "note.md",
                relative_path="note.md",
                content="# Note",
                content_hash="abc",
                note_type="",
                project="",
            )
            client = FakeMemoraClient()
            with patch.dict(
                os.environ,
                {
                    "OPENCODE_API_KEY": "",
                    "MEMORA_EMBEDDING_MODEL": EMBEDDINGGEMMA2_MODEL,
                },
            ):
                with patch(
                    "amai_memory.service.create_client", return_value=client
                ) as create:
                    build_index(paths, [document])
                    query_mode = object()
                    memory_module = types.ModuleType("memora.core.memory")
                    memory_module.QueryMode = SimpleNamespace(
                        BOTH=object(),
                        ORIGINAL=query_mode,
                    )
                    with patch.dict(
                        sys.modules,
                        {
                            "memora": types.ModuleType("memora"),
                            "memora.core": types.ModuleType("memora.core"),
                            "memora.core.memory": memory_module,
                        },
                    ):
                        query_index(paths, "Where is the note?", top_k=3)

            self.assertEqual(client.added_content, ["# Note"])
            question, top_k, _ = client.queries[0]
            self.assertEqual(
                question,
                format_query("Where is the note?", EMBEDDINGGEMMA2_MODEL),
            )
            self.assertEqual(top_k, 3)
            self.assertIs(client.queries[0][2]["query_mode"], query_mode)
            self.assertFalse(create.call_args_list[-1].kwargs["require_api_key"])

    def test_queries_skip_cues_when_cue_index_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = MemoryPaths(vault=root / "vault", data=root / "data")
            paths.data.mkdir(parents=True)
            paths.manifest.write_text(
                json.dumps(
                    {
                        "persist_path": str(root / "persist"),
                        "configuration": {"cue_index": False},
                    }
                ),
                encoding="utf-8",
            )
            client = FakeMemoraClient()
            query_mode = object()
            memory_module = types.ModuleType("memora.core.memory")
            memory_module.QueryMode = SimpleNamespace(
                BOTH=object(),
                ORIGINAL=query_mode,
            )
            with (
                patch(
                    "amai_memory.service.create_client",
                    return_value=client,
                ),
                patch.dict(
                    sys.modules,
                    {
                        "memora": types.ModuleType("memora"),
                        "memora.core": types.ModuleType("memora.core"),
                        "memora.core.memory": memory_module,
                    },
                ),
            ):
                query_index(paths, "primary question")

            self.assertIs(client.queries[0][2]["query_mode"], query_mode)

    def test_cue_results_keep_positive_scores_and_order(self):
        client = FakeMemoraClient(
            [
                SimpleNamespace(index="memory-high", value="high memory", score=0.9),
                SimpleNamespace(index="memory-low", value="low memory", score=0.4),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = MemoryPaths(vault=root / "vault", data=root / "data")
            paths.data.mkdir(parents=True)
            paths.manifest.write_text(
                json.dumps({"persist_path": str(root / "persist")}),
                encoding="utf-8",
            )
            with patch("amai_memory.service.create_client", return_value=client):
                results = query_index(paths, "cue question", top_k=2)

        scores = [result["score"] for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(score > 0.0 for score in scores))

    def test_invalid_update_decision_becomes_new_memory(self):
        _load_memora()
        from memora.builder.memory_builder import MemoryBuilder
        from memora.utils.memory import MemoryUpdateDecision

        class TestBuilder(MemoryBuilder):
            def generate_memory_entries(self, *args, **kwargs):
                return []

            def generate_episodic_memory(self, *args, **kwargs):
                return None

        candidate = SimpleNamespace(
            score=0.9,
            index="existing",
            value="existing value",
            creation_time="now",
        )
        entry = SimpleNamespace(index="new", value="new value")

        for llm_decision in (
            MemoryUpdateDecision(should_update=True),
            MemoryUpdateDecision(should_update=True, best_candidate_index=0),
        ):
            builder = TestBuilder.__new__(TestBuilder)
            builder._model_client = SimpleNamespace(
                invoke=lambda decision=llm_decision, **kwargs: decision
            )
            decision = builder._decide_memory_update(entry, [candidate])

            self.assertFalse(decision["should_update"])
            self.assertIsNone(decision["best_candidate"])


if __name__ == "__main__":
    unittest.main()
