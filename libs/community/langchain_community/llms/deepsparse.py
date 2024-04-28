# flake8: noqa
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import asyncio
from deepsparse import Pipeline
from langchain_core.pydantic_v1 import root_validator
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_community.llms.utils import enforce_stop_tokens

class DeepSparse(LLM):
    """Neural Magic DeepSparse LLM interface.
    To use, you should have the ``deepsparse`` or ``deepsparse-nightly``
    python package installed. See https://github.com/neuralmagic/deepsparse
    This interface let's you deploy optimized LLMs straight from the
    [SparseZoo](https://sparsezoo.neuralmagic.com/?useCase=text_generation)
    Example:
        .. code-block:: python
            from langchain_community.llms import DeepSparse
            llm = DeepSparse(model="zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base_quant-none")
    """  # noqa: E501

    pipeline: Any  #: :meta private:

    model: str
    """The path to a model file or directory or the name of a SparseZoo model stub."""

    model_config: Optional[Dict[str, Any]] = None
    """Keyword arguments passed to the pipeline construction.
    Common parameters are sequence_length, prompt_sequence_length"""

    generation_config: Union[None, str, Dict] = None
    """GenerationConfig dictionary consisting of parameters used to control
    sequences generated for each prompt. Common parameters are:
    max_length, max_new_tokens, num_return_sequences, output_scores,
    top_p, top_k, repetition_penalty."""

    streaming: bool = False
    """Whether to stream the results, token by token."""

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "model_config": self.model_config,
            "generation_config": self.generation_config,
            "streaming": self.streaming,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "deepsparse"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that ``deepsparse`` package is installed."""
        try:
            from deepsparse import Pipeline
        except ImportError:
            raise ImportError(
                "Could not import `deepsparse` package. "
                "Please install it with `pip install deepsparse[llm]`"
            )

        model_config = values["model_config"] or {}

        values["pipeline"] = Pipeline.create(
            task="text_generation",
            model_path=values["model"],
            **model_config,
        )
        return values

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._call, prompt, stop, run_manager, **kwargs
        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        return self._astream(prompt, stop, run_manager, **kwargs)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        task = self.pipeline(sequences=prompt, **self.generation_config)

        if self.streaming:
            for token in task:
                chunk = GenerationChunk(text=token.generations[0].text)
                yield chunk

                if run_manager:
                    run_manager.on_llm_new_token(token=chunk.text)
                    await run_manager.on_llm_new_token.wait()
        else:
            text = (task.generations[0].text)
            if stop is not None:
                text = enforce_stop_tokens(text, stop)
            yield GenerationChunk(text=text)
