from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from groq import Groq
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from selfcheckgpt.prompt_utils import (
    PromptCacheConfig,
    load_cached_json,
    store_cached_json,
)

class SelfCheckAPIPrompt:
    """
    SelfCheckGPT (LLM Prompt): Checking LLM's text against its own sampled texts via API-based prompting (e.g., OpenAI's GPT)
    """
    def __init__(
        self,
        client_type = "openai",
        model = "gpt-3.5-turbo",
        api_key = None,
        **prompt_kwargs,
    ):
        if client_type == "openai":
            # using default keys
            # os.environ.get("OPENAI_ORGANIZATION")
            # os.environ.get("OPENAI_API_KEY")
            self.client = OpenAI()
            print("Initiate OpenAI client... model = {}".format(model)) 
        elif client_type == "groq":
            self.client = Groq(api_key=api_key)
            print("Initiate Groq client... model = {}".format(model))
        
        self.client_type = client_type
        self.model = model
        self.cache_config = PromptCacheConfig.from_kwargs(
            cache_dir=prompt_kwargs.get("cache_dir"),
            use_cache=prompt_kwargs.get("use_cache", False),
        )
        self.prompt_batch_size = prompt_kwargs.get("prompt_batch_size")
        self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()


    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    def completion(self, prompt: str):
        if self.client_type == "openai" or self.client_type == "groq":
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, # 0.0 = deterministic,
                max_tokens=5, # max_tokens is the generated one,
            )
            return chat_completion.choices[0].message.content

        else:
            raise ValueError("client_type not implemented")

    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False,
        **prompt_kwargs,
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :param verson: bool -- if True tqdm progress bar will be shown
        :return sent_scores: sentence-level scores
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose
        cache_config = self.cache_config.override(
            cache_dir=prompt_kwargs.get("cache_dir"),
            use_cache=prompt_kwargs.get("use_cache"),
        )
        batch_size = prompt_kwargs.get("prompt_batch_size", self.prompt_batch_size)
        if batch_size is not None and batch_size < 1:
            batch_size = None
        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]
            pending_jobs = []
            for sample_i, sample in enumerate(sampled_passages):
                sample_clean = sample.replace("\n", " ") 
                prompt = self.prompt_template.format(context=sample_clean, sentence=sentence)
                payload = self._cache_payload(prompt, sentence, sample_clean, sent_i, sample_i)
                cached_entry = load_cached_json(cache_config, payload)
                if cached_entry is not None and "score" in cached_entry:
                    scores[sent_i, sample_i] = cached_entry["score"]
                    continue
                pending_jobs.append((sent_i, sample_i, prompt, payload))
            self._dispatch_api_jobs(pending_jobs, cache_config, batch_size, scores)
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence

    def text_postprocessing(
        self,
        text,
    ):
        """
        To map from generated text to score
        Yes -> 0.0
        No  -> 1.0
        everything else -> 0.5
        """
        text = text.lower().strip()
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]

    def _cache_payload(
        self,
        prompt: str,
        sentence: str,
        sample: str,
        sentence_idx: int,
        sample_idx: int,
    ) -> Dict[str, Any]:
        return {
            "scorer": self.__class__.__name__,
            "client_type": self.client_type,
            "model": self.model,
            "sentence_idx": sentence_idx,
            "sample_idx": sample_idx,
            "prompt": prompt,
            "sentence": sentence,
            "sample": sample,
        }

    def _dispatch_api_jobs(
        self,
        jobs: List[Tuple[int, int, str, Dict[str, Any]]],
        cache_config: PromptCacheConfig,
        batch_size: Optional[int],
        scores: np.ndarray,
    ) -> None:
        if not jobs:
            return
        concurrency = batch_size if batch_size and batch_size > 1 else 1
        if concurrency == 1:
            for job in jobs:
                self._complete_api_job(job, cache_config, scores)
            return
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_job = {executor.submit(self.completion, job[2]): job for job in jobs}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    response = future.result()
                except Exception as exc:
                    print(f"warning: prompt request failed: {exc}")
                    response = "n/a"
                self._finalize_api_job(job, response, cache_config, scores)

    def _complete_api_job(
        self,
        job: Tuple[int, int, str, Dict[str, Any]],
        cache_config: PromptCacheConfig,
        scores: np.ndarray,
    ) -> None:
        try:
            response = self.completion(job[2])
        except Exception as exc:
            print(f"warning: prompt request failed: {exc}")
            response = "n/a"
        self._finalize_api_job(job, response, cache_config, scores)

    def _finalize_api_job(
        self,
        job: Tuple[int, int, str, Dict[str, Any]],
        response: str,
        cache_config: PromptCacheConfig,
        scores: np.ndarray,
    ) -> None:
        sent_i, sample_i, _, payload = job
        score_ = self.text_postprocessing(response)
        scores[sent_i, sample_i] = score_
        store_cached_json(
            cache_config,
            payload,
            {
                "score": float(score_),
                "raw": response,
            },
        )
