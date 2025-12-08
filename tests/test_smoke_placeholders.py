import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

import selfcheckgpt.modeling_selfcheck as modeling_selfcheck


if "openai" not in sys.modules:
    openai_stub = ModuleType("openai")

    class _DefaultOpenAI:  # pragma: no cover - best-effort fallback
        def __init__(self, *args, **kwargs):
            raise RuntimeError("openai client not provided")

    openai_stub.OpenAI = _DefaultOpenAI
    sys.modules["openai"] = openai_stub

if "groq" not in sys.modules:
    groq_stub = ModuleType("groq")

    class _DefaultGroq:  # pragma: no cover - best-effort fallback
        def __init__(self, *args, **kwargs):
            raise RuntimeError("groq client not provided")

    groq_stub.Groq = _DefaultGroq
    sys.modules["groq"] = groq_stub

import selfcheckgpt.modeling_selfcheck_apiprompt as api_prompt_module


def test_mqag_cpu_smoke(monkeypatch):
    questions = [{"question": "Q?", "options": ["a", "b", "c", "d"]}]

    def fake_generation(*args, **kwargs):
        return questions

    def fake_answering(model, tokenizer, question, options, context, max_seq_length, device):
        if context == "passage":
            return np.array([0.6, 0.2, 0.1, 0.1])
        return np.array([0.1, 0.6, 0.2, 0.1])

    def fake_answerability(model, tokenizer, question, context, max_length, device):
        return 0.9 if context == "passage" else 0.4

    monkeypatch.setattr(modeling_selfcheck, "question_generation_sentence_level", fake_generation)
    monkeypatch.setattr(modeling_selfcheck, "answering", fake_answering)
    monkeypatch.setattr(modeling_selfcheck, "answerability_scoring", fake_answerability)

    mqag = modeling_selfcheck.SelfCheckMQAG.__new__(modeling_selfcheck.SelfCheckMQAG)
    mqag.g1_model = mqag.g1_tokenizer = mqag.g2_model = mqag.g2_tokenizer = object()
    mqag.a_model = mqag.a_tokenizer = mqag.u_model = mqag.u_tokenizer = object()
    mqag.device = torch.device("cpu")

    scores = mqag.predict(
        sentences=["Sentence"],
        passage="passage",
        sampled_passages=["sample"],
        scoring_method="bayes_with_alpha",
        beta1=0.8,
        beta2=0.7,
    )
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)
    assert 0.0 <= scores[0] <= 1.0


def test_bertscore_cpu_smoke(monkeypatch):
    class DummySent:
        def __init__(self, text):
            self.text = text

        def __len__(self):
            return len(self.text)

    class DummyDoc:
        def __init__(self, text):
            splits = [s.strip() for s in text.split(".") if s.strip()]
            self._sents = [DummySent(sentence + ".") for sentence in splits]

        @property
        def sents(self):
            return self._sents

    class DummyNLP:
        def __call__(self, text):
            return DummyDoc(text)

    def fake_spacy_load(*args, **kwargs):
        return DummyNLP()

    def fake_bert_score(cands, refs, lang, verbose, rescale_with_baseline):
        size = len(cands)
        tensor = torch.full((size,), 0.75, dtype=torch.float32)
        return tensor, tensor, tensor

    monkeypatch.setattr(modeling_selfcheck.spacy, "load", fake_spacy_load)
    monkeypatch.setattr(modeling_selfcheck.bert_score, "score", fake_bert_score)

    scorer = modeling_selfcheck.SelfCheckBERTScore(default_model="en", rescale_with_baseline=False)
    scores = scorer.predict(
        sentences=["Alpha sentence.", "Beta statement."],
        sampled_passages=["Alpha sentence. Beta statement."],
    )
    assert scores.shape == (2,)
    np.testing.assert_allclose(scores, np.full(2, 0.25))


def test_ngram_cpu_smoke(monkeypatch):
    class DummyNgram:
        def __init__(self, *args, **kwargs):
            self.corpus = []

        def add(self, text):
            self.corpus.append(text)

        def train(self, k=0):
            self.trained = True

        def evaluate(self, sentences):
            base = float(len(self.corpus))
            return {
                "sent_level": {
                    "avg_neg_logprob": [base + idx for idx, _ in enumerate(sentences)],
                    "max_neg_logprob": [base for _ in sentences],
                },
                "doc_level": {
                    "avg_neg_logprob": base,
                    "avg_max_neg_logprob": base + 0.5,
                },
            }

    monkeypatch.setattr(modeling_selfcheck, "UnigramModel", DummyNgram)
    monkeypatch.setattr(modeling_selfcheck, "NgramModel", DummyNgram)

    scorer = modeling_selfcheck.SelfCheckNgram(n=2)
    scores = scorer.predict(
        sentences=["Synthetic sentence."],
        passage="Main passage.",
        sampled_passages=["Sample passage."],
    )
    assert "doc_level" in scores
    assert scores["doc_level"]["avg_neg_logprob"] == pytest.approx(2.0)


def test_nli_cpu_smoke(monkeypatch):
    class DummyBatch(dict):
        def to(self, device):
            return self

    class DummyTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def batch_encode_plus(self, *args, **kwargs):
            return DummyBatch(
                input_ids=torch.zeros((1, 4), dtype=torch.long),
                attention_mask=torch.ones((1, 4), dtype=torch.long),
                token_type_ids=torch.zeros((1, 4), dtype=torch.long),
            )

    class DummyModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def eval(self):
            return self

        def to(self, device):
            self.device = device
            return self

        def __call__(self, **kwargs):
            logits = torch.tensor([[0.2, 0.8]], dtype=torch.float32)
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(modeling_selfcheck, "DebertaV2Tokenizer", DummyTokenizer)
    monkeypatch.setattr(modeling_selfcheck, "DebertaV2ForSequenceClassification", DummyModel)

    scorer = modeling_selfcheck.SelfCheckNLI()
    scores = scorer.predict(sentences=["Premise"], sampled_passages=["Hypothesis"])
    assert scores.shape == (1,)
    expected = torch.softmax(torch.tensor([[0.2, 0.8]], dtype=torch.float32), dim=-1)[0, 1].item()
    assert scores[0] == pytest.approx(expected)


def test_llm_prompt_cpu_smoke(monkeypatch):
    class DummyInputs(dict):
        def __init__(self, input_ids):
            super().__init__(input_ids=input_ids)
            self.input_ids = input_ids

        def to(self, device):
            return self

    class DummyTokenizer:
        def __init__(self):
            self.last_prompt = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, prompt, return_tensors=None, padding=None):
            self.last_prompt = prompt
            if isinstance(prompt, list):
                batch = torch.ones((len(prompt), 3), dtype=torch.long)
            else:
                batch = torch.ones((1, 3), dtype=torch.long)
            return DummyInputs(batch)

        def batch_decode(self, generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            prompts = self.last_prompt
            if isinstance(prompts, list):
                return [f"{prompt} Yes" for prompt in prompts]
            return [f"{prompts} Yes"]

    class DummyModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def eval(self):
            return self

        def to(self, device):
            return self

        def generate(self, input_ids=None, max_new_tokens=None, do_sample=None, **kwargs):
            if input_ids is None:
                input_ids = torch.tensor([[0]])
            return input_ids

    monkeypatch.setattr(modeling_selfcheck, "AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr(modeling_selfcheck, "AutoModelForCausalLM", DummyModel)

    scorer = modeling_selfcheck.SelfCheckLLMPrompt(model="dummy")
    scores = scorer.predict(
        sentences=["Check me."],
        sampled_passages=["Context here."],
    )
    assert scores.shape == (1,)
    assert scores[0] == pytest.approx(0.0)


def test_api_prompt_cpu_smoke(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create),
            )

        def _create(self, model, messages, temperature, max_tokens):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Yes"))]
            )

    monkeypatch.setattr(api_prompt_module, "OpenAI", lambda *args, **kwargs: DummyClient())

    scorer = api_prompt_module.SelfCheckAPIPrompt(client_type="openai", model="dummy")
    scores = scorer.predict(
        sentences=["API check."],
        sampled_passages=["Context snippet."],
    )
    assert scores.shape == (1,)
    assert scores[0] == pytest.approx(0.0)
