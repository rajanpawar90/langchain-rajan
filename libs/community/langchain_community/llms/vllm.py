from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import Field, root_validator

from langchain_community.llms.openai import BaseOpenAI
from langchain_community.utils.openai import is_openai_v1

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "Could not import vllm python package. "
        "Please install it with `pip install vllm`."
    )


class VLLM(BaseLLM):
    """VLLM language model."""

    model: str = ""
    tensor_parallel_size: Optional[int] = 1
    trust_remote_code: Optional[bool] = False
    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    use_beam_search: bool = False
    stop: Optional[List[str]] = None
    ignore_eos: bool = False
    max_new_tokens: int = 512
    logprobs: Optional[int] = None
    dtype: str = "auto"
    download_dir: Optional[str] = None
    vllm_kwargs: Dict[str, Any] = Field(default_factory=dict)

    client: Any

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        values["client"] = vllm.LLM(
            model=values["model"],
            tensor_parallel_size=values["tensor_parallel_size"],
            trust_remote_code=values["trust_remote_code"],
            dtype=values["dtype"],
            download_dir=values["download_dir"],
            **values["vllm_kwargs"],
        )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling vllm."""
        return {
            "n": self.n,
            "best_of": self.best_of,
            "max_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop": self.stop,
            "ignore_eos": self.ignore_eos,
            "use_beam_search": self.use_beam_search,
            "logprobs": self.logprobs,
        }

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""

        params = {**self._default_params, **kwargs, "stop": stop}
        outputs = self.client.generate(prompts, params)

        generations = []
        for output in outputs:
            text = output.outputs[0].text
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "vllm"


class VLLMOpenAI(BaseOpenAI):
    """vLLM OpenAI-compatible API client"""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""

        params = {
            "model": self.model_name,
            "logit_bias": None,
        }

        if not is_openai_v1():
            params.update(
                {
                    "api_key": self.openai_api_key,
                    "api_base": self.openai_api_base,
                }
            )

        return params

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "vllm-openai"
