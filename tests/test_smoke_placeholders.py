import pytest


@pytest.mark.skip(reason="TODO: add CPU-only smoke for MQAG variant without heavy downloads")
def test_mqag_placeholder():
    pass


@pytest.mark.skip(reason="TODO: add CPU-only smoke for BERTScore variant with lightweight inputs")
def test_bertscore_placeholder():
    pass


@pytest.mark.skip(reason="TODO: add CPU-only smoke for Ngram variant with tiny corpus")
def test_ngram_placeholder():
    pass


@pytest.mark.skip(reason="TODO: add CPU-only smoke for NLI variant with mocked tokenizer/model")
def test_nli_placeholder():
    pass


@pytest.mark.skip(reason="TODO: add CPU-only smoke for prompt-based variants with stubbed responses")
def test_prompt_placeholder():
    pass
