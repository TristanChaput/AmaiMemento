from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import local
from typing import Any, Callable, Iterable, Sequence

from amai_memory.memora_adapter import (
    EMBEDDINGGEMMA2_MODEL,
    QWEN_EMBEDDING_MODEL,
    configured_embedding_model,
    create_client,
    cue_index_enabled,
    embedding_spec,
    format_query,
)
from amai_memory.vault import VaultDocument, iter_vault_documents


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REBUILD_WORKERS = 2


@dataclass(frozen=True)
class MemoryPaths:
    vault: Path
    data: Path

    @classmethod
    def from_environment(cls) -> "MemoryPaths":
        vault = Path(
            os.getenv("AMAI_KNOWLEDGE_PATH", REPOSITORY_ROOT.parent / "AmaiKnowledge")
        )
        data = Path(os.getenv("MEMORA_DATA_PATH", REPOSITORY_ROOT / "data" / "memora"))
        return cls(vault=vault.resolve(), data=data.resolve())

    @property
    def manifest(self) -> Path:
        return self.data / "current.json"


def scan(paths: MemoryPaths) -> list[VaultDocument]:
    return list(iter_vault_documents(paths.vault))


def _document_snapshot(documents: list[VaultDocument]) -> list[dict[str, str]]:
    return [
        {"path": document.relative_path, "hash": document.content_hash}
        for document in documents
    ]


def _write_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(checkpoint, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _find_resume_checkpoint(
    paths: MemoryPaths,
    documents: list[VaultDocument],
    resume_after: str,
    enable_cue_index: bool,
    embedding_model: str,
) -> tuple[Path, dict]:
    expected_documents = _document_snapshot(documents)
    active_build_id = (current_manifest(paths) or {}).get("build_id")
    checkpoint_paths = sorted(
        (paths.data / "builds").glob("*/checkpoint.json"),
        reverse=True,
    )
    for checkpoint_path in checkpoint_paths:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        completed_count = checkpoint.get("completed_count", 0)
        if (
            checkpoint.get("last_completed") != resume_after
            or completed_count <= 0
            or completed_count > len(documents)
            or checkpoint.get("build_id") == active_build_id
        ):
            continue
        if documents[completed_count - 1].relative_path != resume_after:
            raise ValueError("The resume note does not match the checkpoint position.")
        if checkpoint.get("documents") != expected_documents:
            raise ValueError("The vault changed since this interrupted rebuild.")
        if checkpoint.get("vault_path") != str(paths.vault):
            raise ValueError("This checkpoint belongs to a different vault.")
        if checkpoint.get("configuration", {}).get("cue_index") != enable_cue_index:
            raise ValueError("The cue-index setting differs from this checkpoint.")
        checkpoint_model = checkpoint.get("configuration", {}).get("embedding_model")
        if checkpoint_model is not None and checkpoint_model != embedding_model:
            raise ValueError("The embedding model differs from this checkpoint.")
        return checkpoint_path, checkpoint
    raise ValueError(
        f"No resumable rebuild found after {resume_after!r}; "
        "only builds with saved checkpoints can resume."
    )


def current_manifest(paths: MemoryPaths) -> dict | None:
    if not paths.manifest.is_file():
        return None
    return json.loads(paths.manifest.read_text(encoding="utf-8"))


def _resolve_rebuild_workers(requested: int | None, document_count: int) -> int:
    """Resolve the number of extraction workers without exceeding the workload."""
    if requested is None:
        raw_value = os.getenv("MEMORA_REBUILD_WORKERS", str(DEFAULT_REBUILD_WORKERS))
        try:
            requested = int(raw_value)
        except ValueError as error:
            raise ValueError(
                "MEMORA_REBUILD_WORKERS must be a positive integer."
            ) from error
    if requested < 1:
        raise ValueError("Rebuild workers must be at least 1.")
    return min(requested, max(document_count, 1))


class _ParallelDocumentExtractor:
    """Extract document memories in worker-local OpenCode sessions.

    Memora's extraction is independent per document, but its upsert/update path
    is not transactional. This class therefore only performs the remote LLM
    phase; callers must persist the returned entries sequentially.
    """

    def __init__(self, cfg: Any, agent_memory: Any, enable_cue_index: bool):
        self.cfg = cfg
        self.agent_memory = agent_memory
        self.enable_cue_index = enable_cue_index
        self._thread_state = local()

    def _builder(self) -> Any:
        builder = getattr(self._thread_state, "builder", None)
        if builder is None:
            from amai_memory.memora_adapter import OpenCodeChatCompletionModel
            from memora.builder.document_memory_builder import DocumentMemoryBuilder

            model_client = OpenCodeChatCompletionModel(self.cfg)
            builder = DocumentMemoryBuilder(
                self.cfg,
                self.agent_memory,
                model_client,
            )
            self._thread_state.builder = builder
        return builder

    def __call__(self, document: VaultDocument) -> list[Any]:
        from memora.utils.memory import generate_metadata
        from memora.utils.misc import normalize_content

        multimodal_support = bool(self.cfg.memory.get("multimodal_support", False))
        content = normalize_content(
            document.content,
            multimodal_support=multimodal_support,
        )
        metadata = generate_metadata(content, dict(document.metadata))
        return self._builder().generate_memory_entries(
            content,
            metadata,
            enable_cue_index=self.enable_cue_index,
        )


def _parallel_rebuild_components(client: Any) -> tuple[Any, Any, Any] | None:
    """Return the local Memora internals needed for staged extraction.

    The fallback to the normal public ``client.add`` path keeps tests and
    future Memora clients compatible if these local implementation details
    change.
    """
    local_client = getattr(client, "_client", None)
    if local_client is None:
        return None
    cfg = getattr(local_client, "cfg", None)
    agent_memory = getattr(local_client, "_agent_memory", None)
    model_client = getattr(local_client, "_model_client", None)
    registry = getattr(local_client, "memory_builder_registry", None)
    if any(value is None for value in (cfg, agent_memory, model_client, registry)):
        return None
    if bool(cfg.memory.get("enable_episodic_memory", False)):
        return None
    storage_builder = registry.get(
        "markdown",
        cfg,
        agent_memory,
        model_client,
    )
    return cfg, agent_memory, storage_builder


def _upsert_entries(
    storage_builder: Any,
    agent_memory: Any,
    entries: list[Any],
) -> set[str]:
    """Persist entries and report their final indices without scanning Chroma.

    Memora can replace an extracted index while merging a memory. Tracking its
    ``add`` calls captures that final index and avoids listing the full growing
    collection after every source document.
    """
    changed_indices = {entry.index for entry in entries if entry.index}
    original_add = getattr(agent_memory, "add", None)
    instance_namespace = getattr(agent_memory, "__dict__", None)
    had_instance_add = (
        isinstance(instance_namespace, dict) and "add" in instance_namespace
    )
    original_instance_add = instance_namespace.get("add") if had_instance_add else None
    tracking_installed = False

    if callable(original_add):

        def tracked_add(entry: Any) -> Any:
            result = original_add(entry)
            if entry.index:
                changed_indices.add(entry.index)
            return result

        try:
            setattr(agent_memory, "add", tracked_add)
            tracking_installed = True
        except (AttributeError, TypeError):
            pass

    try:
        for entry in entries:
            storage_builder.upsert_memory_entry(entry=entry)
    finally:
        if tracking_installed:
            if had_instance_add:
                setattr(agent_memory, "add", original_instance_add)
            else:
                delattr(agent_memory, "add")

    return changed_indices


def build_index(
    paths: MemoryPaths,
    documents: Iterable[VaultDocument],
    progress: Callable[[int, int, str], None] | None = None,
    workers: int | None = None,
    enable_cue_index: bool | None = None,
    resume_after: str | None = None,
) -> dict:
    documents = list(documents)
    workers = _resolve_rebuild_workers(workers, len(documents))
    if enable_cue_index is None:
        enable_cue_index = cue_index_enabled()
    embedding_model = configured_embedding_model()
    if resume_after is not None:
        resume_after = resume_after.replace("\\", "/")
        checkpoint_path, checkpoint = _find_resume_checkpoint(
            paths,
            documents,
            resume_after,
            enable_cue_index,
            embedding_model,
        )
        build_id = checkpoint["build_id"]
        persist_path = checkpoint_path.parent
        completed_count = checkpoint["completed_count"]
        memory_count = checkpoint["memory_count"]
        provenance = checkpoint["provenance"]
    else:
        build_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        persist_path = paths.data / "builds" / build_id
        if persist_path.exists():
            build_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            persist_path = paths.data / "builds" / build_id
        completed_count = 0
        memory_count = 0
        provenance: dict[str, list[dict[str, str]]] = {}

    persist_path.mkdir(parents=True, exist_ok=True)
    client = create_client(
        persist_path,
        enable_cue_index=enable_cue_index,
        embedding_model=embedding_model,
    )
    started_at = time.perf_counter()
    previous_values: dict[str, str] = {}
    parallel_components = _parallel_rebuild_components(client) if workers > 1 else None
    parallel_enabled = parallel_components is not None
    if resume_after is not None and not parallel_enabled:
        previous_values = {
            entry.index: getattr(entry, "val", getattr(entry, "value", ""))
            for entry in client.list_memories(limit=100_000)
        }

    checkpoint_path = persist_path / "checkpoint.json"
    checkpoint = {
        "build_id": build_id,
        "vault_path": str(paths.vault),
        "documents": _document_snapshot(documents),
        "configuration": {
            "cue_index": enable_cue_index,
            "embedding_model": embedding_model,
        },
        "completed_count": completed_count,
        "last_completed": (
            documents[completed_count - 1].relative_path if completed_count else None
        ),
        "memory_count": memory_count,
        "provenance": provenance,
    }
    _write_checkpoint(checkpoint_path, checkpoint)

    def persist_document(
        index: int,
        document: VaultDocument,
        entries: list[Any],
        changed_indices: set[str] | None = None,
    ) -> None:
        nonlocal completed_count, memory_count, previous_values
        memory_count += len(entries)
        if changed_indices is None:
            current_memories = client.list_memories(limit=100_000)
            current_values = {
                entry.index: getattr(entry, "val", getattr(entry, "value", ""))
                for entry in current_memories
            }
            changed_indices = {
                memory_index
                for memory_index, value in current_values.items()
                if previous_values.get(memory_index) != value
            }
            changed_indices.update(entry.index for entry in entries if entry.index)
            previous_values = current_values
        for memory_index in changed_indices:
            sources = provenance.setdefault(memory_index, [])
            if document.metadata not in sources:
                sources.append(document.metadata)
        completed_count = index
        checkpoint.update(
            completed_count=completed_count,
            last_completed=document.relative_path,
            memory_count=memory_count,
            provenance=provenance,
        )
        _write_checkpoint(checkpoint_path, checkpoint)
        if progress:
            progress(index, len(documents), document.relative_path)

    if parallel_enabled:
        cfg, agent_memory, storage_builder = parallel_components
        storage_builder._build_stats = {
            "new": 0,
            "update": 0,
            "duplicate": 0,
            "extracted": 0,
        }
        extractor = _ParallelDocumentExtractor(
            cfg,
            agent_memory,
            enable_cue_index=enable_cue_index,
        )
        first_pending_index = completed_count
        remaining_documents = documents[completed_count:]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for batch_start in range(0, len(remaining_documents), workers):
                batch = remaining_documents[batch_start : batch_start + workers]
                extracted_documents = executor.map(extractor, batch)
                for index, (document, entries) in enumerate(
                    zip(batch, extracted_documents),
                    first_pending_index + batch_start + 1,
                ):
                    changed_indices = _upsert_entries(
                        storage_builder,
                        agent_memory,
                        entries,
                    )
                    persist_document(
                        index,
                        document,
                        entries,
                        changed_indices=changed_indices,
                    )
    else:
        for index, document in enumerate(
            documents[completed_count:], completed_count + 1
        ):
            entries = client.add(
                document.content,
                type="markdown",
                metadata=document.metadata,
            )
            persist_document(index, document, entries)

    provenance_path = persist_path / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "build_id": build_id,
        "built_at": datetime.now(UTC).isoformat(),
        "persist_path": str(persist_path),
        "vault_path": str(paths.vault),
        "document_count": len(documents),
        "memory_count": memory_count,
        "provenance_path": str(provenance_path),
        "configuration": {
            "cue_index": enable_cue_index,
            "embedding_model": embedding_model,
        },
        "rebuild": {
            "workers": workers,
            "parallel_extraction": parallel_enabled,
            "duration_seconds": round(time.perf_counter() - started_at, 3),
        },
    }
    paths.data.mkdir(parents=True, exist_ok=True)
    temporary_manifest = paths.data / "current.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(paths.manifest)
    return manifest


def query_index(paths: MemoryPaths, question: str, top_k: int = 5) -> list[dict]:
    manifest = current_manifest(paths)
    if not manifest:
        raise FileNotFoundError(
            "No active Memora index. Run the rebuild command first."
        )
    embedding_model = (
        manifest.get("configuration", {}).get("embedding_model")
        or configured_embedding_model()
    )
    # Memora prints diagnostics to stdout; keep the CLI output machine-readable.
    with redirect_stdout(sys.stderr):
        client = create_client(
            Path(manifest["persist_path"]),
            require_api_key=False,
            embedding_model=embedding_model,
        )
    query = format_query(question, embedding_model)
    try:
        from memora.core.memory import QueryMode
    except ModuleNotFoundError:
        results = client.query(query, top_k=top_k)
    else:
        build_configuration = manifest.get("configuration", {})
        # Manifests created before this field existed used cues by default.
        build_has_cues = build_configuration.get("cue_index", True)
        # Without cues, search all types; PRIMARY_ONLY hardcodes lowercase "factual".
        query_mode = QueryMode.BOTH if build_has_cues else QueryMode.ORIGINAL
        results = client.query(query, top_k=top_k, query_mode=query_mode)
    provenance_path = Path(manifest.get("provenance_path", ""))
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else {}
    )
    return [
        {
            "index": entry.index,
            "value": entry.value,
            "score": entry.score,
            "sources": provenance.get(entry.index, []),
        }
        for entry in results
    ]


def _process_rss_mb() -> float | None:
    """Best-effort resident set size of the current process, in MiB."""
    try:
        import psutil  # type: ignore

        return round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    except Exception:
        pass
    try:
        import resource  # type: ignore

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _Counters()
            counters.cb = ctypes.sizeof(counters)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )
            return round(counters.WorkingSetSize / 1024 / 1024, 1)
        except Exception:
            return None
    return None


def _load_index_memories(persist_path: Path) -> list[tuple[str, str]]:
    """Read (embedded document, value) pairs directly from the Chroma index.

    This avoids building a Memora client, so the index's own embedding model is
    never loaded just to list the memories it contains.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_path))
    collections = client.list_collections()
    names = [
        collection
        if isinstance(collection, str)
        else getattr(collection, "name", str(collection))
        for collection in collections
    ]
    if not names:
        return []
    preferred = next(
        (name for name in names if name.startswith("amai_knowledge")), names[0]
    )
    collection = client.get_collection(preferred)
    result = collection.get(include=["documents", "metadatas"], limit=100_000)
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    return [
        (document, (metadata or {}).get("value", ""))
        for document, metadata in zip(documents, metadatas)
        if document
    ]


def _benchmark_worker(payload: dict) -> dict:
    """Encode and rank memories for one model, inside its own process.

    Running each model in a fresh process keeps its memory footprint isolated
    and measurable: the resident set is captured before load, after load, and
    after encoding.
    """
    import numpy as np

    from amai_memory.memora_adapter import format_query, load_embedding_model

    name = payload["model"]
    documents = payload["documents"]
    questions = payload["questions"]
    values = payload["values"]
    top_k = payload["top_k"]

    baseline_rss = _process_rss_mb()
    model = load_embedding_model(name)
    spec = model.spec
    loaded_rss = _process_rss_mb()

    started = time.perf_counter()
    doc_vectors = model.generate_embeddings(documents)
    document_seconds = time.perf_counter() - started

    matrix = np.asarray(doc_vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12

    per_query: list[dict[str, Any]] = []
    rankings: list[list[int]] = []
    started = time.perf_counter()
    for question in questions:
        vector = np.asarray(
            model.generate_embeddings([format_query(question, name)])[0],
            dtype=np.float32,
        )
        vector /= np.linalg.norm(vector) + 1e-12
        scores = matrix @ vector
        order = np.argsort(-scores)[:top_k]
        rankings.append([int(position) for position in order])
        per_query.append(
            {
                "question": question,
                "results": [
                    {
                        "index": documents[position],
                        "value": values.get(documents[position], ""),
                        "score": round(float(scores[position]), 4),
                    }
                    for position in order
                ],
            }
        )
    query_seconds = time.perf_counter() - started

    return {
        "dimension": spec.dimension,
        "text_only": spec.text_only,
        "document_seconds": round(document_seconds, 3),
        "query_seconds": round(query_seconds, 3),
        "per_query_seconds": round(query_seconds / len(questions), 4),
        "baseline_rss_mb": baseline_rss,
        "model_rss_mb": (
            round(loaded_rss - baseline_rss, 1)
            if loaded_rss is not None and baseline_rss is not None
            else None
        ),
        "total_rss_mb": _process_rss_mb(),
        "retrieval": per_query,
        "rankings": rankings,
    }


def benchmark_embeddings(
    paths: MemoryPaths,
    questions: Sequence[str] | None = None,
    top_k: int = 5,
    models: Sequence[str] | None = None,
    probe_count: int = 8,
) -> dict:
    """Compare candidate embedding models over the active index's memories.

    The memories were extracted once by the LLM and are reused as-is, so only
    the embedding model varies. Each model runs in its own process, giving
    isolated latency and memory numbers, without another rebuild and without
    touching the active index.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    manifest = current_manifest(paths)
    if not manifest:
        raise FileNotFoundError(
            "No active Memora index. Run the rebuild command first."
        )

    model_names = (
        list(models) if models else [QWEN_EMBEDDING_MODEL, EMBEDDINGGEMMA2_MODEL]
    )
    for name in model_names:
        embedding_spec(name)

    index_model = (
        manifest.get("configuration", {}).get("embedding_model")
        or configured_embedding_model()
    )
    memories = _load_index_memories(Path(manifest["persist_path"]))
    documents = [document for document, _ in memories]
    values = {document: value for document, value in memories}
    if not documents:
        raise ValueError("The active index contains no memories to benchmark.")

    if not questions:
        stride = max(len(documents) // max(probe_count, 1), 1)
        questions = documents[::stride][:probe_count]
    questions = list(questions)
    if not questions:
        raise ValueError("At least one benchmark question is required.")

    results: dict[str, Any] = {
        "index": {
            "build_id": manifest.get("build_id"),
            "embedding_model": index_model,
            "memory_count": len(documents),
        },
        "questions": questions,
        "top_k": top_k,
        "models": {},
    }

    context = multiprocessing.get_context("spawn")
    rankings: dict[str, list[list[int]]] = {}
    for name in model_names:
        payload = {
            "model": name,
            "documents": documents,
            "questions": questions,
            "values": values,
            "top_k": top_k,
        }
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            model_result = executor.submit(_benchmark_worker, payload).result()
        rankings[name] = model_result.pop("rankings")
        results["models"][name] = model_result

    reference = model_names[0]
    comparison = []
    for question_index, question in enumerate(questions):
        reference_top = rankings[reference][question_index]
        entry: dict[str, Any] = {
            "question": question,
            "overlap": {},
            "top1_agreement": {},
        }
        for name in model_names[1:]:
            other_top = rankings[name][question_index]
            depth = min(top_k, len(reference_top), len(other_top))
            overlap = (
                len(set(reference_top[:depth]) & set(other_top[:depth])) / depth
                if depth
                else 0.0
            )
            entry["overlap"][name] = round(overlap, 3)
            entry["top1_agreement"][name] = bool(
                reference_top and other_top and reference_top[0] == other_top[0]
            )
        comparison.append(entry)

    summary = {}
    for name in model_names[1:]:
        overlaps = [row["overlap"][name] for row in comparison]
        top1 = [row["top1_agreement"][name] for row in comparison]
        summary[name] = {
            "mean_overlap_at_k": (
                round(sum(overlaps) / len(overlaps), 3) if overlaps else 0.0
            ),
            "top1_agreement_rate": (round(sum(top1) / len(top1), 3) if top1 else 0.0),
        }

    results["reference"] = reference
    results["comparison"] = comparison
    results["summary"] = summary
    return results
