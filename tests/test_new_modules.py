"""
Unit tests for newly added modules: decomposition, orchestrator, data, prompt_utils.
These tests run without external dependencies (no model downloads, no API calls).
"""

import json
import tempfile
from pathlib import Path
from typing import List

import pytest

from selfcheckgpt.decomposition import simple_clause_split, _heuristic_split
from selfcheckgpt.orchestrator import score_with_decomposition, _coerce_float_list, _resolve_aggregation
from selfcheckgpt.prompt_utils import (
    PromptCacheConfig,
    make_cache_key,
    load_cached_json,
    store_cached_json,
)
from selfcheckgpt import data as data_module


# ============================================================================
# Decomposition tests
# ============================================================================

class TestDecomposition:
    def test_simple_clause_split_empty(self):
        assert simple_clause_split("") == []
        assert simple_clause_split("   ") == []

    def test_simple_clause_split_single_clause(self):
        result = simple_clause_split("Hello world.")
        assert len(result) >= 1
        assert "Hello" in result[0] or "world" in result[0]

    def test_simple_clause_split_multiple_clauses(self):
        sentence = "Paris is beautiful, and Rome is ancient."
        result = simple_clause_split(sentence)
        # Should split on comma or conjunction
        assert len(result) >= 1

    def test_heuristic_split_with_punctuation(self):
        sentence = "First part, second part; third part."
        result = _heuristic_split(sentence)
        assert len(result) >= 2

    def test_heuristic_split_with_conjunctions(self):
        sentence = "Cats are nice and dogs are loyal"
        result = _heuristic_split(sentence)
        assert len(result) >= 1


# ============================================================================
# Orchestrator tests
# ============================================================================

class DummyScorer:
    """Toy scorer for testing orchestrator."""
    def predict(self, sentences: List[str], sampled_passages: List[str], **kwargs) -> List[float]:
        return [len(s) / 100.0 for s in sentences]


class TestOrchestrator:
    def test_score_without_decomposition(self):
        scorer = DummyScorer()
        sentences = ["Short.", "A longer sentence here."]
        result = score_with_decomposition(
            scorer=scorer,
            sentences=sentences,
            sampled_passages=["sample"],
            use_decomposition=False,
        )
        assert "scores" in result
        assert len(result["scores"]) == 2
        assert result["aggregation"] is None

    def test_score_with_decomposition_mean(self):
        scorer = DummyScorer()
        sentences = ["First clause, second clause."]
        result = score_with_decomposition(
            scorer=scorer,
            sentences=sentences,
            sampled_passages=["sample"],
            use_decomposition=True,
            aggregation="mean",
        )
        assert "scores" in result
        assert "chunks_by_sentence" in result
        assert result["aggregation"] == "mean"

    def test_score_with_decomposition_max(self):
        scorer = DummyScorer()
        sentences = ["First clause, second clause."]
        result = score_with_decomposition(
            scorer=scorer,
            sentences=sentences,
            sampled_passages=["sample"],
            use_decomposition=True,
            aggregation="max",
        )
        assert result["aggregation"] == "max"

    def test_coerce_float_list_from_list(self):
        assert _coerce_float_list([1, 2, 3]) == [1.0, 2.0, 3.0]

    def test_coerce_float_list_from_single(self):
        assert _coerce_float_list(5.5) == [5.5]

    def test_resolve_aggregation_mean(self):
        fn, label = _resolve_aggregation("mean")
        assert label == "mean"
        assert fn([1, 2, 3]) == 2.0

    def test_resolve_aggregation_max(self):
        fn, label = _resolve_aggregation("max")
        assert label == "max"
        assert fn([1, 2, 3]) == 3.0

    def test_resolve_aggregation_invalid(self):
        with pytest.raises(ValueError):
            _resolve_aggregation("invalid")


# ============================================================================
# Prompt utils (caching) tests
# ============================================================================

class TestPromptUtils:
    def test_cache_config_from_kwargs(self):
        config = PromptCacheConfig.from_kwargs(cache_dir="/tmp/test", use_cache=True)
        assert config.enabled is True
        assert config.cache_dir == Path("/tmp/test")

    def test_cache_config_override(self):
        config = PromptCacheConfig.from_kwargs(cache_dir="/tmp/a", use_cache=False)
        overridden = config.override(cache_dir="/tmp/b", use_cache=True)
        assert overridden.cache_dir == Path("/tmp/b")
        assert overridden.enabled is True

    def test_make_cache_key_deterministic(self):
        payload = {"a": 1, "b": "hello"}
        key1 = make_cache_key(payload)
        key2 = make_cache_key(payload)
        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex digest

    def test_make_cache_key_different_for_different_payloads(self):
        key1 = make_cache_key({"a": 1})
        key2 = make_cache_key({"a": 2})
        assert key1 != key2

    def test_store_and_load_cached_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PromptCacheConfig(cache_dir=Path(tmpdir), enabled=True)
            payload = {"prompt": "test", "idx": 0}
            value = {"score": 0.5, "raw": "Yes"}

            store_cached_json(config, payload, value)
            loaded = load_cached_json(config, payload)

            assert loaded is not None
            assert loaded["score"] == 0.5

    def test_load_cached_json_disabled(self):
        config = PromptCacheConfig(cache_dir=Path("/tmp"), enabled=False)
        result = load_cached_json(config, {"any": "payload"})
        assert result is None


# ============================================================================
# Data loader tests
# ============================================================================

class TestDataLoader:
    def test_list_loaders(self):
        loaders = data_module.list_loaders()
        assert "events" in loaders
        assert "places" in loaders
        assert "organizations" in loaders

    def test_get_loader_unknown(self):
        with pytest.raises(KeyError):
            data_module.get_loader("nonexistent_domain")

    def test_load_events_dev(self):
        examples = data_module.load("events", split="dev", limit=1)
        assert len(examples) == 1
        assert "id" in examples[0]
        assert "prompt" in examples[0]
        assert "reference" in examples[0]
        assert "samples" in examples[0]

    def test_load_places_dev(self):
        examples = data_module.load("places", split="dev", limit=1)
        assert len(examples) == 1

    def test_load_organizations_dev(self):
        examples = data_module.load("organizations", split="dev", limit=1)
        assert len(examples) == 1

    def test_configure_loader_override(self):
        # Test that configure_loader doesn't crash
        data_module.configure_loader(
            "events",
            local_path="/nonexistent/path",
        )
        # Reset by loading normally (will use default path)
        data_module._LOADER_OVERRIDES.pop("events", None)

