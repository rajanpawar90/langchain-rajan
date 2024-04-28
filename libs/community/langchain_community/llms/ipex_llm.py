import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Literal, Mapping, Optional

import transformers
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra

DEFAULT_MODEL_ID = "gpt2"

logger = logging.getLogger(__name__)

class IpexLLM(LLM):
    """IpexLLM model.

    Example:
        .. code-block:: python

            from langchain_community.llms import IpexLLM
            llm = IpexLLM.from_model_id(model_id="THUDM/chatglm-6b")
    """

    model_id: str = DEFAULT_MODEL_ID
    """Model name or model path to use."""
    model_kwargs: Optional[Dict[str, Any]] = None
    """Keyword arguments passed to the model."""
    model: Any = None  #: :meta private:
    """IpexLLM model."""
    tokenizer: Any = None  #: :meta private:
    """Huggingface tokenizer model."""
    streaming: bool = True
    """Whether to stream the results, token by token."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.model = None
        cls.tokenizer = None

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "IpexLLM":
        """
        Construct object from model_id

        Args:
            model_id: Path for the huggingface repo id to be downloaded or
                      the huggingface checkpoint folder.
            model_kwargs: Keyword arguments to pass to the model and tokenizer.
            kwargs: Extra arguments to pass to the model and tokenizer.

        Returns:
            An object of IpexLLM.
        """
        try:
            from ipex_llm.transformers import (
                AutoModel,
                AutoModelForCausalLM,
            )
        except ImportError:
            if isinstance(os.geterror(), OSError):
                raise ValueError(
                    "Could not import ipex-llm or transformers. "
                    "Please install it with `pip install --pre --upgrade ipex-llm[all]`."
                )
            else:
                raise

        _model_kwargs = model_kwargs or {}

        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
        except Exception:
            cls.tokenizer = transformers.LlamaTokenizer.from_pretrained(model_id, **_model_kwargs)

        try:
            cls.model = AutoModelForCausalLM.from_pretrained(
                model_id, load_in_4bit=True, **_model_kwargs
            )
        except Exception:
            cls.model = AutoModel.from_pretrained(
                model_id, load_in_4bit=True, **_model_kwargs
            )

        if "trust_remote_code" in _model_kwargs:
            _model_kwargs = {
                k: v for k, v in _model_kwargs.items() if k != "trust_remote_code"
            }

        self = cls(
            model_id=model_id,
            model_kwargs=_model_kwargs,
            **kwargs,
        )
        self._identifying_params.update(
            {
                "model_id": self.model_id,
                "model_kwargs": self.model_kwargs,
            }
        )
        return self

    @classmethod
    def from_model_id_low_bit(
        cls,
        model_id: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "IpexLLM":
        """
        Construct low_bit object from model_id

        Args:

            model_id: Path for the ipex-llm transformers low-bit model folder.
            model_kwargs: Keyword arguments to pass to the model and tokenizer.
            kwargs: Extra arguments to pass to the model and tokenizer.

        Returns:
            An object of IpexLLM.
        """
        try:
            from ipex_llm.transformers import (
                AutoModel,
                AutoModelForCausalLM,
            )
        except ImportError:
            if isinstance(os.geterror(), OSError):
                raise ValueError(
                    "Could not import ipex-llm or transformers. "
                    "Please install it with `pip install --pre --upgrade ipex-llm[all]`."
                )
            else:
                raise

        _model_kwargs = model_kwargs or {}
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
        except Exception:
            cls.tokenizer = transformers.LlamaTokenizer.from_pretrained(model_id, **_model_kwargs)

        try:
            cls.model = AutoModelForCausalLM.load_low_bit(model_id, **_model_kwargs)
        except Exception:
            cls.model = AutoModel.load_low_bit(model_id, **_model_kwargs)

        if "trust_remote_code" in _model_kwargs:
            _model_kwargs = {
                k: v for k, v in _model_kwargs.items() if k != "trust_remote_code"
            }

        self = cls(
            model_id=model_id,
            model_kwargs=_model_kwargs,
            **kwargs,
        )
        self._identifying_params.update(
            {
                "model_id": self.model_id,
                "model_kwargs": self.model_kwargs,
            }
        )
        return self

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "ipex-llm"

    @lru_cache(maxsize=None)
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.streaming is True:
            from transformers import TextStreamer

            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            streamer = TextStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            if stop is not None:
                from transformers.generation.stopping_criteria import (
                    StoppingCriteriaList,
                )
                from transformers.tools.agents import StopSequenceCriteria

                # stop generation when stop words are encountered
                # TODO: stop generation when the following one is stop word
                stopping_criteria = StoppingCriteriaList(
                    [StopSequenceCriteria(stop, self.tokenizer)]
                )
            else:
                stopping_criteria = None
            output = self.model.generate(
                input_ids,
                streamer=streamer,
                stopping_criteria=stopping_criteria,
                **kwargs,
            )
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return text
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            if stop is not None:
                from transformers.generation.stopping_criteria import (
                    StoppingCriteriaList,
                )
                from transformers.tools.agents import StopSequenceCriteria

                stopping_criteria = StoppingCriteriaList(
                    [StopSequenceCriteria(stop, self.tokenizer)]
                )
            else:
                stopping_criteria = None
            output = self.model.generate(
                input_ids, stopping_criteria=stopping_criteria, **kwargs
            )
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)[
                len(prompt) :
            ]
            return text
