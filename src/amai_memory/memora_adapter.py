from __future__ import annotations

import json
import os
import sys
import uuid
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from threading import RLock
from typing import Any, Sequence, get_origin

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MEMORA_SOURCE = REPOSITORY_ROOT / "vendor" / "Memora" / "src"
OPENCODE_BASE_URL = "https://opencode.ai/zen/go/v1"
OPENCODE_USER_AGENT = "amai-memory/0.1.0"
QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_EMBEDDING_DIMENSION = 1024
EMBEDDINGGEMMA2_MODEL = "google/embeddinggemma-2"
EMBEDDINGGEMMA2_DIMENSION = 768
DEFAULT_EMBEDDING_MODEL = EMBEDDINGGEMMA2_MODEL
DEFAULT_EMBEDDING_CACHE_SIZE = 8192
QWEN_QUERY_PREFIX = (
    "Instruct: Given a query about a personal knowledge base, retrieve relevant memories that answer it\n"
    "Query: "
)
# Private-use marker stripped by the embedding model to switch to the query
# prompt. It is prepended by ``format_query`` and never appears in vault text.
QUERY_PROMPT_SENTINEL = "\ue000amai-query\ue000"


@dataclass(frozen=True)
class EmbeddingSpec:
    """A supported local embedding model and how to encode text for it."""

    model_name: str
    dimension: int
    query_prefix: str = ""
    query_prompt_name: str | None = None
    document_prompt_name: str | None = None
    text_only: bool = False


EMBEDDING_SPECS: dict[str, EmbeddingSpec] = {
    QWEN_EMBEDDING_MODEL: EmbeddingSpec(
        model_name=QWEN_EMBEDDING_MODEL,
        dimension=QWEN_EMBEDDING_DIMENSION,
        query_prefix=QWEN_QUERY_PREFIX,
    ),
    EMBEDDINGGEMMA2_MODEL: EmbeddingSpec(
        model_name=EMBEDDINGGEMMA2_MODEL,
        dimension=EMBEDDINGGEMMA2_DIMENSION,
        query_prefix=QUERY_PROMPT_SENTINEL,
        query_prompt_name="SearchQuery",
        document_prompt_name="Document",
        text_only=True,
    ),
}
load_dotenv(REPOSITORY_ROOT / ".env")


def embedding_spec(model_name: str | None = None) -> EmbeddingSpec:
    """Return the spec for a model, defaulting to the configured one."""
    resolved = model_name or configured_embedding_model()
    try:
        return EMBEDDING_SPECS[resolved]
    except KeyError:
        supported = ", ".join(sorted(EMBEDDING_SPECS))
        raise ValueError(
            f"Unsupported embedding model {resolved!r}. Supported models: {supported}."
        ) from None


def configured_embedding_model() -> str:
    """Return the embedding model selected through the environment."""
    return os.getenv("MEMORA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


_embedding_cache: OrderedDict[tuple[str, str], array] = OrderedDict()
_embedding_cache_lock = RLock()


class MemoraUnavailableError(RuntimeError):
    """Raised when the optional Memora runtime is not installed or configured."""


def _config_value(cfg: Any, section: str, key: str, default: Any = None) -> Any:
    values = getattr(cfg, section, None)
    if values is None:
        return default
    getter = getattr(values, "get", None)
    return getter(key, default) if getter else getattr(values, key, default)


def _environment_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean (true/false).")


def cue_index_enabled() -> bool:
    return _environment_bool("MEMORA_ENABLE_CUE_INDEX", False)


def _embedding_cache_size() -> int:
    raw_value = os.getenv(
        "MEMORA_EMBEDDING_CACHE_SIZE",
        str(DEFAULT_EMBEDDING_CACHE_SIZE),
    )
    try:
        size = int(raw_value)
    except ValueError as error:
        raise ValueError(
            "MEMORA_EMBEDDING_CACHE_SIZE must be a non-negative integer."
        ) from error
    if size < 0:
        raise ValueError("MEMORA_EMBEDDING_CACHE_SIZE must be a non-negative integer.")
    return size


def _clear_embedding_cache() -> None:
    """Clear the process-local cache (mainly useful for isolated tests)."""
    with _embedding_cache_lock:
        _embedding_cache.clear()


def _validated_embedding(vector: Any, dimension: int) -> array:
    values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    embedding = array("f", (float(value) for value in values))
    if len(embedding) != dimension:
        raise ValueError(
            f"Expected {dimension}-dimensional embeddings, got {len(embedding)}."
        )
    return embedding


@cache
def _load_sentence_transformer(model_name: str = QWEN_EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


@cache
def _load_text_only_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name,
        device="cpu",
        config_kwargs={"vision_config": None, "audio_config": None},
    )


def load_embedding_model(model_name: str | None = None) -> "LocalEmbeddingModel":
    """Load a supported embedding model without a full Memora configuration."""
    from types import SimpleNamespace

    name = model_name or configured_embedding_model()
    return LocalEmbeddingModel(SimpleNamespace(openai={"embedding_model": name}))


class LocalEmbeddingModel:
    """Memora's embedding interface backed by a cached local model.

    Supports Qwen (default) and EmbeddingGemma 2. Documents and queries are
    encoded asymmetrically: queries carry a task prefix (Qwen's instruction or
    EmbeddingGemma's private-use sentinel mapped to its ``SearchQuery`` prompt),
    documents use the model's ``Document`` prompt when one is defined.
    """

    def __init__(self, cfg: Any, client: Any = None):
        self.cfg = cfg
        self.client = client
        self.model_name = _config_value(
            cfg, "openai", "embedding_model", configured_embedding_model()
        )
        self.spec = embedding_spec(self.model_name)
        if self.spec.text_only:
            self.model = _load_text_only_sentence_transformer(self.model_name)
        else:
            self.model = _load_sentence_transformer(self.model_name)

    def get_client(self) -> Any:
        return self.client

    def _prepare(self, text: str, prompt_name: str | None) -> tuple[str | None, str]:
        """Resolve the sentence-transformers prompt and the text to encode."""
        if prompt_name is not None:
            return prompt_name, text
        if self.spec.query_prompt_name and text.startswith(self.spec.query_prefix):
            return (
                self.spec.query_prompt_name,
                text[len(self.spec.query_prefix) :],
            )
        return self.spec.document_prompt_name, text

    def generate_embeddings(
        self,
        input: Sequence[str],
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        if isinstance(input, str):
            raise TypeError("Embedding input must be a list of strings.")
        texts = list(input)
        if not texts:
            return []

        cache_size = _embedding_cache_size()
        keys = [(self.model_name, text) for text in texts]
        with _embedding_cache_lock:
            resolved: dict[tuple[str, str], array] = {}
            missing_keys = []
            missing_texts = []
            seen_missing = set()
            for key, text in zip(keys, texts):
                if cache_size and key in _embedding_cache:
                    _embedding_cache.move_to_end(key)
                    resolved[key] = _embedding_cache[key]
                elif key not in seen_missing:
                    seen_missing.add(key)
                    missing_keys.append(key)
                    missing_texts.append(text)

            generated: dict[tuple[str, str], array] = {}
            if missing_texts:
                prepared = [
                    (key, *self._prepare(text, prompt_name))
                    for key, text in zip(missing_keys, missing_texts)
                ]
                # Encode one prompt group at a time (Chroma calls are homogeneous
                # but a caller may batch queries and documents together).
                groups: dict[str | None, list[tuple[Any, str]]] = {}
                order: list[tuple[str | None, Any]] = []
                for key, group_prompt, encoded_text in prepared:
                    if group_prompt not in groups:
                        groups[group_prompt] = []
                        order.append((group_prompt, key))
                    groups[group_prompt].append((key, encoded_text))
                for group_prompt, _ in order:
                    items = groups[group_prompt]
                    group_texts = [item[1] for item in items]
                    if group_prompt is None:
                        vectors = self.model.encode(
                            group_texts,
                            normalize_embeddings=True,
                        )
                    else:
                        vectors = self.model.encode(
                            group_texts,
                            normalize_embeddings=True,
                            prompt_name=group_prompt,
                        )
                    if len(vectors) != len(group_texts):
                        raise ValueError(
                            "The embedding model returned an unexpected number "
                            "of vectors."
                        )
                    for (key, _), vector in zip(items, vectors):
                        generated[key] = _validated_embedding(
                            vector, self.spec.dimension
                        )
                resolved.update(generated)
                if cache_size:
                    _embedding_cache.update(generated)
                    while len(_embedding_cache) > cache_size:
                        _embedding_cache.popitem(last=False)

            return [list(resolved[key]) for key in keys]


# Backwards-compatible name for the previous Qwen-only implementation.
QwenEmbeddingModel = LocalEmbeddingModel


def format_query(question: str, model_name: str | None = None) -> str:
    spec = embedding_spec(model_name)
    return f"{spec.query_prefix}{question}"


def format_qwen_query(question: str) -> str:
    return format_query(question, QWEN_EMBEDDING_MODEL)


def _create_openai_client(api_key: str) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=OPENCODE_BASE_URL,
        default_headers={
            "User-Agent": OPENCODE_USER_AGENT,
            "x-opencode-session": f"amai-memory-{uuid.uuid4().hex}",
        },
    )


def _messages(input: str | Sequence[dict[str, Any]], prompt_args: dict | None) -> list:
    if isinstance(input, str):
        content = input.format(**prompt_args) if prompt_args else input
        return [{"role": "user", "content": content}]
    if isinstance(input, Sequence):
        return list(input)
    raise ValueError("Input must be a string or a sequence of messages.")


def _parse_structured_response(response_format: Any, content: str) -> Any:
    try:
        return response_format.model_validate_json(content)
    except ValueError as error:
        last_error = error

    decoder = json.JSONDecoder()
    skip_until = 0
    for start, character in enumerate(content):
        if start < skip_until or character not in "[{":
            continue
        try:
            payload, end = decoder.raw_decode(content[start:])
        except ValueError:
            continue
        skip_until = start + end
        try:
            return response_format.model_validate(payload)
        except ValueError as error:
            last_error = error
            fields = response_format.model_fields
            if len(fields) != 1 or not isinstance(payload, (dict, list)):
                continue
            field_name, field = next(iter(fields.items()))
            if get_origin(field.annotation) is not list or (
                isinstance(payload, dict) and field_name in payload
            ):
                continue
            try:
                items = payload if isinstance(payload, list) else [payload]
                return response_format.model_validate({field_name: items})
            except ValueError as error:
                last_error = error
    raise last_error


def _structured_messages(messages: list, response_format: Any) -> list:
    schema = json.dumps(response_format.model_json_schema(), ensure_ascii=False)
    instruction = (
        "Return only valid JSON. Do not use Markdown fences or explanatory text. "
        "The JSON must match this schema:\n"
        f"{schema}"
    )
    return [{"role": "system", "content": instruction}, *messages]


def _is_response_format_unavailable(error: Exception) -> bool:
    message = str(error).lower()
    return "response_format" in message and (
        "unavailable" in message or "not supported" in message
    )


class OpenCodeChatCompletionModel:
    """OpenCode Go client for Chat Completions and GPT Responses models."""

    def __init__(self, cfg: Any, token_usage_callback: Any = None):
        self.cfg = cfg
        self.token_usage_callback = token_usage_callback
        self.model_type = "opencode"
        self._json_mode_available: bool | None = None
        api_key = _config_value(cfg, "openai", "api_key")
        self.client = _create_openai_client(api_key) if api_key else None

    def _complete(self, request: dict) -> tuple[str | None, ...]:
        if request["model"].startswith("gpt-"):
            request = dict(request)
            request["input"] = request.pop("messages")
            request.pop("seed", None)
            if "response_format" in request:
                request["text"] = {"format": request.pop("response_format")}
            response = self.client.responses.create(**request, store=False)
            return (response.output_text,)
        response = self.client.chat.completions.create(**request)
        message = response.choices[0].message
        return (message.content, getattr(message, "reasoning_content", None))

    def invoke(
        self,
        input: str | Sequence[dict[str, Any]],
        prompt_args: dict | None = None,
        response_format: Any = None,
        source: str = "Unknown",
        **kwargs: Any,
    ) -> Any:
        if self.client is None:
            raise MemoraUnavailableError(
                "OPENCODE_API_KEY is required for memory extraction."
            )

        request = {
            "messages": _messages(input, prompt_args),
            "model": _config_value(self.cfg, "llm", "model", "glm-5.2"),
            "seed": _config_value(self.cfg, "llm", "seed", 42),
            **kwargs,
        }
        if response_format is None:
            return self._complete(request)[0]

        last_error = None
        for _ in range(2):
            structured_request = {
                **request,
                "messages": _structured_messages(request["messages"], response_format),
            }
            if self._json_mode_available is not False:
                structured_request["response_format"] = {"type": "json_object"}
            try:
                contents = self._complete(structured_request)
                self._json_mode_available = True
            except Exception as error:
                if (
                    self._json_mode_available is False
                    or not _is_response_format_unavailable(error)
                ):
                    raise
                self._json_mode_available = False
                structured_request.pop("response_format", None)
                contents = self._complete(structured_request)
            for content in contents:
                if not content:
                    continue
                try:
                    return _parse_structured_response(response_format, content)
                except ValueError as error:
                    last_error = error
            if not any(contents):
                last_error = ValueError(
                    f"{response_format.__name__} response is empty."
                )
        raise last_error


class _OfflineChatCompletionModel:
    """Placeholder used by semantic queries; it never creates a remote client."""

    def __init__(self, cfg: Any, token_usage_callback: Any = None):
        self.cfg = cfg
        self.token_usage_callback = token_usage_callback
        self.model_type = "offline"
        self.client = None

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        raise MemoraUnavailableError(
            "LLM calls are disabled during local memory queries."
        )


class _CompatibleChatCompletionModel:
    def __new__(cls, cfg: Any, token_usage_callback: Any = None):
        if _config_value(cfg, "openai", "api_key"):
            return OpenCodeChatCompletionModel(cfg, token_usage_callback)
        return _OfflineChatCompletionModel(cfg, token_usage_callback)


def _install_compatibility() -> None:
    import memora.utils.embedding as embedding_module
    import memora.utils.llm as llm_module

    vendor_chat_model = llm_module.ChatCompletionModel
    vendor_embedding_model = embedding_module.BaseEmbeddingModel
    llm_module.ChatCompletionModel = _CompatibleChatCompletionModel
    embedding_module.BaseEmbeddingModel = LocalEmbeddingModel

    for module in tuple(sys.modules.values()):
        namespace = getattr(module, "__dict__", {})
        if namespace.get("ChatCompletionModel") is vendor_chat_model:
            namespace["ChatCompletionModel"] = _CompatibleChatCompletionModel
        if namespace.get("BaseEmbeddingModel") is vendor_embedding_model:
            namespace["BaseEmbeddingModel"] = LocalEmbeddingModel


def _load_memora() -> tuple[Any, Any]:
    if not (MEMORA_SOURCE / "memora" / "memora_client.py").is_file():
        raise MemoraUnavailableError(
            "Memora submodule is missing. Run: git submodule update --init vendor/Memora"
        )
    source = str(MEMORA_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        from omegaconf import OmegaConf

        _install_compatibility()
        from memora.memora_client import MemoraClient
    except ImportError as error:
        raise MemoraUnavailableError(
            "Memora dependencies are missing. Install vendor/Memora/requirements.txt "
            "in a Python 3.12 environment."
        ) from error
    return MemoraClient, OmegaConf


def create_client(
    persist_path: Path,
    require_api_key: bool = True,
    enable_cue_index: bool | None = None,
    embedding_model: str | None = None,
):
    api_key = os.getenv("OPENCODE_API_KEY") if require_api_key else None
    if require_api_key and not api_key:
        raise MemoraUnavailableError("OPENCODE_API_KEY is required for Memora rebuild.")

    MemoraClient, OmegaConf = _load_memora()
    if enable_cue_index is None:
        enable_cue_index = cue_index_enabled()
    llm_model = os.getenv("MEMORA_LLM_MODEL", "glm-5.2")
    embedding_model = embedding_model or configured_embedding_model()
    embedding_spec(embedding_model)
    config = OmegaConf.create(
        {
            "llm": {"model": llm_model, "seed": 42},
            "openai": {
                "api_type": "openai",
                "api_key": api_key,
                "embedding_model": embedding_model,
                "model": llm_model,
            },
            "memory": {
                "memory_store": "amai_knowledge",
                "persist_path": str(persist_path),
                "collection_name": "amai_knowledge",
                "distance": "cosine",
                "query_score_threshold": 0.35,
                "update_score_threshold": 0.8,
                "force_rebuild": False,
                "enhance_query": False,
                "return_history": True,
                "multimodal_support": False,
                "top_k": 10,
                "cue_top_k": 10,
                "enable_hybrid_search": False,
                "enable_segmentation": False,
                "enable_episodic_memory": False,
                "use_segments_as_episodic": False,
                "enable_cue_index": enable_cue_index,
            },
            "retrieval": {"strategy": "semantic"},
            "eval": {"max_workers": 1},
        }
    )
    return MemoraClient(cfg=config, user_id="amai_knowledge")
