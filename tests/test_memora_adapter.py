import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import BaseModel

from amai_memory.memora_adapter import (
    EMBEDDINGGEMMA2_DIMENSION,
    EMBEDDINGGEMMA2_MODEL,
    MemoraUnavailableError,
    OpenCodeChatCompletionModel,
    QWEN_EMBEDDING_DIMENSION,
    QWEN_EMBEDDING_MODEL,
    QUERY_PROMPT_SENTINEL,
    LocalEmbeddingModel,
    QwenEmbeddingModel,
    _clear_embedding_cache,
    _create_openai_client,
    _load_sentence_transformer,
    _load_text_only_sentence_transformer,
    _parse_structured_response,
    create_client,
    embedding_spec,
    format_query,
)


class FakeStructuredResponse(BaseModel):
    value: str


class FakeMemoryOutput(BaseModel):
    memory_type: str
    index: str
    value: str


class FakeMemoryOutputs(BaseModel):
    entries: list[FakeMemoryOutput]


class FakeVector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeSentenceTransformer:
    instances = []

    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.encode_calls = []
        self.instances.append(self)

    def encode(self, texts, normalize_embeddings):
        self.encode_calls.append((list(texts), normalize_embeddings))
        return [FakeVector([1.0] * QWEN_EMBEDDING_DIMENSION) for _ in texts]


class FakePromptSentenceTransformer:
    instances = []

    def __init__(self, model_name, device, config_kwargs=None):
        self.model_name = model_name
        self.device = device
        self.config_kwargs = config_kwargs
        self.encode_calls = []
        self.instances.append(self)

    def encode(self, texts, normalize_embeddings, prompt_name=None):
        self.encode_calls.append((list(texts), normalize_embeddings, prompt_name))
        return [FakeVector([1.0] * EMBEDDINGGEMMA2_DIMENSION) for _ in texts]


def config(api_key=None, embedding_model=QWEN_EMBEDDING_MODEL):
    return SimpleNamespace(
        llm=SimpleNamespace(model="glm-5.2", seed=42),
        openai={"api_key": api_key, "embedding_model": embedding_model},
    )


class MemoraAdapterTests(unittest.TestCase):
    def setUp(self):
        FakeSentenceTransformer.instances.clear()
        FakePromptSentenceTransformer.instances.clear()
        _load_sentence_transformer.cache_clear()
        _load_text_only_sentence_transformer.cache_clear()
        _clear_embedding_cache()

    def tearDown(self):
        _load_sentence_transformer.cache_clear()
        _load_text_only_sentence_transformer.cache_clear()
        _clear_embedding_cache()

    def test_unwrapped_memory_payloads_are_wrapped_in_entries(self):
        memory = (
            '{"memory_type":"Factual","index":"settings","value":"language settings"}'
        )

        for content in (memory, f"[{memory},{memory}]"):
            result = _parse_structured_response(FakeMemoryOutputs, content)
            self.assertTrue(all(entry.index == "settings" for entry in result.entries))

    def test_invalid_entries_report_the_root_payload_error(self):
        content = (
            '{"entries":['
            '{"memory_type":"Factual","index":"missing value"},'
            '{"memory_type":"Factual","index":"valid","value":"kept"}'
            "]}"
        )

        with self.assertRaisesRegex(ValueError, r"entries\.0\.value"):
            _parse_structured_response(FakeMemoryOutputs, content)

    def test_opencode_client_uses_json_mode_and_keeps_structured_output(self):
        create = Mock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="1. **Analyze the Request**",
                            reasoning_content='` reasoning tokens: {"value": "from reasoning"} `',
                        )
                    )
                ]
            )
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with patch(
            "amai_memory.memora_adapter._create_openai_client",
            return_value=fake_client,
        ) as create_openai:
            model = OpenCodeChatCompletionModel(config("test-key"))
            result = model.invoke("hello", response_format=FakeStructuredResponse)

        create_openai.assert_called_once_with("test-key")
        create.assert_called_once()
        self.assertEqual(
            create.call_args.kwargs["response_format"]["type"],
            "json_object",
        )
        self.assertIn(
            "Return only valid JSON", create.call_args.kwargs["messages"][0]["content"]
        )
        self.assertEqual(result.value, "from reasoning")

    def test_gpt_uses_responses_for_plain_and_structured_output(self):
        for structured in (False, True):
            with self.subTest(structured=structured):
                create = Mock(return_value=SimpleNamespace(output_text='{"value":"ok"}'))
                fake_client = SimpleNamespace(responses=SimpleNamespace(create=create))
                cfg = config("test-key")
                cfg.llm.model = "gpt-6-luna"
                with patch(
                    "amai_memory.memora_adapter._create_openai_client",
                    return_value=fake_client,
                ):
                    result = OpenCodeChatCompletionModel(cfg).invoke(
                        "hello",
                        response_format=FakeStructuredResponse if structured else None,
                    )
                self.assertEqual(result.value if structured else result, "ok" if structured else '{"value":"ok"}')
                request = create.call_args.kwargs
                self.assertEqual(request["model"], "gpt-6-luna")
                self.assertEqual(request["input"][-1], {"role": "user", "content": "hello"})
                self.assertFalse(request["store"])
                self.assertNotIn("seed", request)
                self.assertNotIn("messages", request)
                self.assertNotIn("response_format", request)
                if structured:
                    self.assertEqual(request["text"], {"format": {"type": "json_object"}})
                else:
                    self.assertNotIn("text", request)

    def test_gpt_retries_invalid_structured_output_once(self):
        create = Mock(side_effect=[
            SimpleNamespace(output_text='{"value":"cut'),
            SimpleNamespace(output_text='{"value":"complete"}'),
        ])
        cfg = config("test-key")
        cfg.llm.model = "gpt-6-luna"
        with patch(
            "amai_memory.memora_adapter._create_openai_client",
            return_value=SimpleNamespace(responses=SimpleNamespace(create=create)),
        ):
            result = OpenCodeChatCompletionModel(cfg).invoke(
                "hello", response_format=FakeStructuredResponse
            )
        self.assertEqual(result.value, "complete")
        self.assertEqual(create.call_count, 2)

    def test_structured_output_falls_back_when_json_mode_is_unavailable(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"value":"from fallback"}',
                        reasoning_content=None,
                    )
                )
            ]
        )
        create = Mock(
            side_effect=[
                RuntimeError("This response_format type is unavailable now"),
                response,
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with patch(
            "amai_memory.memora_adapter._create_openai_client",
            return_value=fake_client,
        ):
            result = OpenCodeChatCompletionModel(config("test-key")).invoke(
                "hello", response_format=FakeStructuredResponse
            )

        self.assertEqual(result.value, "from fallback")
        self.assertEqual(create.call_count, 2)
        self.assertEqual(
            create.call_args_list[0].kwargs["response_format"],
            {"type": "json_object"},
        )
        self.assertNotIn("response_format", create.call_args_list[1].kwargs)

    def test_truncated_structured_response_is_retried_once(self):
        def response(content, finish_reason):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason=finish_reason,
                        message=SimpleNamespace(
                            content=content, reasoning_content=None
                        ),
                    )
                ]
            )

        create = Mock(
            side_effect=[
                response('{"value":"cut', "length"),
                response('{"value":"complete"}', "stop"),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with patch(
            "amai_memory.memora_adapter._create_openai_client",
            return_value=fake_client,
        ):
            result = OpenCodeChatCompletionModel(config("test-key")).invoke(
                "hello", response_format=FakeStructuredResponse
            )

        self.assertEqual(result.value, "complete")
        self.assertEqual(create.call_count, 2)

    def test_openai_factory_receives_opencode_base_url(self):
        calls = []
        module = types.ModuleType("openai")

        class FakeOpenAI:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        module.OpenAI = FakeOpenAI
        with patch.dict(sys.modules, {"openai": module}):
            _create_openai_client("test-key")

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["api_key"], "test-key")
        self.assertEqual(calls[0]["base_url"], "https://opencode.ai/zen/go/v1")
        self.assertEqual(calls[0]["default_headers"]["User-Agent"], "amai-memory/0.1.0")
        self.assertRegex(
            calls[0]["default_headers"]["x-opencode-session"],
            r"^amai-memory-[0-9a-f]{32}$",
        )

    def test_series_of_queries_builds_sentence_transformer_once(self):
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = FakeSentenceTransformer
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            models = [QwenEmbeddingModel(config()) for _ in range(3)]
            embeddings = [
                model.generate_embeddings([question])
                for model, question in zip(models, ["one", "two", "three"])
            ]

        self.assertTrue(all(model.model is models[0].model for model in models))
        self.assertEqual(len(FakeSentenceTransformer.instances), 1)
        self.assertEqual(
            FakeSentenceTransformer.instances[0].model_name, QWEN_EMBEDDING_MODEL
        )
        self.assertEqual(FakeSentenceTransformer.instances[0].device, "cpu")
        self.assertEqual(
            FakeSentenceTransformer.instances[0].encode_calls,
            [(["one"], True), (["two"], True), (["three"], True)],
        )
        self.assertEqual(len(embeddings), 3)
        self.assertEqual(len(embeddings[0][0]), QWEN_EMBEDDING_DIMENSION)
        self.assertTrue(all(isinstance(value, float) for value in embeddings[0][0]))

    def test_duplicate_embeddings_are_encoded_once_across_instances(self):
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = FakeSentenceTransformer
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            first = QwenEmbeddingModel(config())
            second = QwenEmbeddingModel(config())
            embeddings = first.generate_embeddings(["same", "same", "other"])
            cached = second.generate_embeddings(["same"])

        self.assertEqual(
            FakeSentenceTransformer.instances[0].encode_calls,
            [(["same", "other"], True)],
        )
        self.assertEqual(len(embeddings), 3)
        self.assertEqual(cached[0], embeddings[0])

    def test_small_embedding_cache_still_returns_evicted_batch_hits(self):
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = FakeSentenceTransformer
        with (
            patch.dict(sys.modules, {"sentence_transformers": module}),
            patch.dict(
                os.environ,
                {"MEMORA_EMBEDDING_CACHE_SIZE": "1"},
            ),
        ):
            model = QwenEmbeddingModel(config())
            model.generate_embeddings(["cached"])
            embeddings = model.generate_embeddings(["cached", "new"])

        self.assertEqual(len(embeddings), 2)
        self.assertEqual(
            FakeSentenceTransformer.instances[0].encode_calls,
            [(["cached"], True), (["new"], True)],
        )

    def test_rebuild_requires_opencode_key_before_loading_memora(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"OPENCODE_API_KEY": ""}):
                with self.assertRaises(MemoraUnavailableError):
                    create_client(Path(directory))

    def test_supported_embedding_specs_declare_their_dimensions(self):
        self.assertEqual(embedding_spec(QWEN_EMBEDDING_MODEL).dimension, 1024)
        self.assertEqual(
            embedding_spec(EMBEDDINGGEMMA2_MODEL).dimension, EMBEDDINGGEMMA2_DIMENSION
        )
        self.assertTrue(embedding_spec(EMBEDDINGGEMMA2_MODEL).text_only)

    def test_unsupported_embedding_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported embedding model"):
            embedding_spec("some/unknown-model")

    def test_format_query_is_model_specific(self):
        self.assertTrue(
            format_query("hello", QWEN_EMBEDDING_MODEL).endswith("Query: hello")
        )
        self.assertEqual(
            format_query("hello", EMBEDDINGGEMMA2_MODEL),
            f"{QUERY_PROMPT_SENTINEL}hello",
        )

    def test_embeddinggemma_encodes_documents_and_queries_with_prompts(self):
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = FakePromptSentenceTransformer
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            model = LocalEmbeddingModel(config(embedding_model=EMBEDDINGGEMMA2_MODEL))
            model.generate_embeddings(["a memory"])
            model.generate_embeddings(
                [format_query("a question", EMBEDDINGGEMMA2_MODEL)]
            )

        instance = FakePromptSentenceTransformer.instances[0]
        self.assertEqual(
            instance.config_kwargs, {"vision_config": None, "audio_config": None}
        )
        self.assertEqual(
            instance.encode_calls,
            [
                (["a memory"], True, "Document"),
                (["a question"], True, "SearchQuery"),
            ],
        )

    def test_embeddinggemma_rejects_wrong_dimension(self):
        class WrongDimensionTransformer(FakePromptSentenceTransformer):
            def encode(self, texts, normalize_embeddings, prompt_name=None):
                self.encode_calls.append(
                    (list(texts), normalize_embeddings, prompt_name)
                )
                return [FakeVector([1.0] * 128) for _ in texts]

        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = WrongDimensionTransformer
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            model = LocalEmbeddingModel(config(embedding_model=EMBEDDINGGEMMA2_MODEL))
            with self.assertRaisesRegex(ValueError, "768-dimensional"):
                model.generate_embeddings(["a memory"])


if __name__ == "__main__":
    unittest.main()
