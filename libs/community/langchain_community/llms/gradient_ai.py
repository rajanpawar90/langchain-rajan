import asyncio
import dataclasses
import logging
import os
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
    overload,
)
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, Field, root_validator

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env

T = TypeVar("T")


class TrainResult(TypedDict):
    """Train result."""

    loss: float


@dataclasses.dataclass
class GradientLLMConfig(BaseModel):
    """Configuration for GradientLLM."""

    model_id: str = Field(alias="model", min_length=2)
    """Underlying gradient.ai model id (base or fine-tuned)."""

    gradient_workspace_id: Optional[str] = None
    """Underlying gradient.ai workspace_id."""

    gradient_access_token: Optional[str] = None
    """gradient.ai API Token, which can be generated by going to
    https://auth.gradient.ai/select-workspace
    and selecting "Access tokens" under the profile drop-down.
    """

    model_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model."""

    gradient_api_url: str = "https://api.gradient.ai/api"
    """Endpoint URL to use."""

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Mapping[str, Any]) -> Mapping[str, Any]:
        """Validate that api key and python package exists in environment."""

        values["gradient_access_token"] = get_from_dict_or_env(
            values, "gradient_access_token", "GRADIENT_ACCESS_TOKEN"
        )
        values["gradient_workspace_id"] = get_from_dict_or_env(
            values, "gradient_workspace_id", "GRADIENT_WORKSPACE_ID"
        )

        if (
            values["gradient_access_token"] is None
            or len(values["gradient_access_token"]) < 10
        ):
            raise ValueError("env variable `GRADIENT_ACCESS_TOKEN` must be set")

        if (
            values["gradient_workspace_id"] is None
            or len(values["gradient_access_token"]) < 3
        ):
            raise ValueError("env variable `GRADIENT_WORKSPACE_ID` must be set")

        if values["model_kwargs"]:
            kw = values["model_kwargs"]
            if not 0 <= kw.get("temperature", 0.5) <= 1:
                raise ValueError("`temperature` must be in the range [0.0, 1.0]")

            if not 0 <= kw.get("top_p", 0.5) <= 1:
                raise ValueError("`top_p` must be in the range [0.0, 1.0]")

            if 0 >= kw.get("top_k", 0.5):
                raise ValueError("`top_k` must be positive")

            if 0 >= kw.get("max_generated_token_count", 1):
                raise ValueError("`max_generated_token_count` must be positive")

        values["gradient_api_url"] = get_from_dict_or_env(
            values, "gradient_api_url", "GRADIENT_API_URL"
        )

        try:
            import gradientai  # noqa
        except ImportError:
            logging.warning(
                "DeprecationWarning: `GradientLLM` will use "
                "`pip install gradientai` in future releases of langchain."
            )
        except Exception:
            pass

        return values


class GradientLLM(BaseLLM):
    """Gradient.ai LLM Endpoints.

    GradientLLM is a class to interact with LLMs on gradient.ai

    To use, set the environment variable ``GRADIENT_ACCESS_TOKEN`` with your
    API token and ``GRADIENT_WORKSPACE_ID`` for your gradient workspace,
    or alternatively provide them as keywords to the constructor of this class.

    Example:
        .. code-block:: python

            from langchain_community.llms import GradientLLM
            GradientLLM(
                model="99148c6d-c2a0-4fbe-a4a7-e7c05bdb8a09_base_ml_model",
                model_kwargs={
                    "max_generated_token_count": 128,
                    "temperature": 0.75,
                    "top_p": 0.95,
                    "top_k": 20,
                    "stop": [],
                },
                gradient_workspace_id="12345614fc0_workspace",
                gradient_access_token="gradientai-access_token",
            )

    """

    config: GradientLLMConfig = Field(default_factory=GradientLLMConfig)
    """Configuration for this instance of GradientLLM."""

    aiosession: Optional[httpx.AsyncClient] = None  #: :meta private:
    """ClientSession, private, subject to change in upcoming releases."""

    @dataclasses.dataclass
    class _CallKwargs:
        prompt: str
        stop: Optional[List[str]] = None
        run_manager: Optional[CallbackManagerForLLMRun] = None
        **kwargs: Any

    @overload
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        ...

    @overload
    def _call(
        self,
        prompts: Sequence[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> List[str]:
        ...

    def _call(
        self,
        prompt: Union[str, Sequence[str]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """Call to Gradients API `model/{id}/complete`.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        if not isinstance(prompt, str):
            return [self._call(p, stop, run_manager, **kwargs) for p in prompt]

        kwargs = self._CallKwargs(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs)
        return self._internal_call(**dataclasses.asdict(kwargs))

    async def _acall(
        self,
        prompt: Union[str, Sequence[str]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """Async Call to Gradients API `model/{id}/complete`.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        if not isinstance(prompt, str):
            return await asyncio.gather(
                *(self._acall(p, stop, run_manager, **kwargs) for p in prompt)
            )

        kwargs = self._CallKwargs(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs)
        return await self._internal_acall(**dataclasses.asdict(kwargs))

    @dataclasses.dataclass
    class _InternalCallKwargs(Annotated[_CallKwargs, "InternalCallKwargs"]):
        session: httpx.AsyncClient

    def _internal_call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Internal call to Gradients API `model/{id}/complete`.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        async def _inner_call() -> str:
            return await self._call_inner(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
                session=self.aiosession or httpx.AsyncClient(),
                **kwargs,
            )

        return asyncio.run(_inner_call())

    async def _internal_acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Internal async call to Gradients API `model/{id}/complete`.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        return await self._call_inner(
            prompt=prompt,
            stop=stop,
            run_manager=run_manager,
            session=self.aiosession or httpx.AsyncClient(),
            **kwargs,
        )

    async def _call_inner(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Union[CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun]] = None,
        session: httpx.AsyncClient,
        **kwargs: Any,
    ) -> str:
        """Call to Gradients API `model/{id}/complete`.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        url = urljoin(self.config.gradient_api_url, f"models/{self.config.model_id}/complete")

        headers = {
            "authorization": f"Bearer {self.config.gradient_access_token}",
            "x-gradient-workspace-id": f"{self.config.gradient_workspace_id}",
            "accept": "application/json",
            "content-type": "application/json",
        }

        payload = {
            "query": prompt,
            "maxGeneratedTokenCount": kwargs.get("max_generated_token_count", None),
            "temperature": kwargs.get("temperature", None),
            "topK": kwargs.get("top_k", None),
            "topP": kwargs.get("top_p", None),
        }

        async with session.post(url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                raise Exception(
                    f"Gradient returned an unexpected response with status "
                    f"{response.status}: {await response.text()}"
                )

            text = (await response.json())["generatedOutput"]

            if stop is not None:
                # Apply stop tokens when making calls to Gradient
                text = enforce_stop_tokens(text, stop)

            return text

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""

        generations = [
            Generation(
                text=self._call(
                    prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
                )
            )
            for prompt in prompts
        ]

        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        generations = [
            Generation(text=await self._acall(prompt, stop, run_manager, **kwargs))
            for prompt in prompts
        ]
        return LLMResult(generations=generations)

    def train_unsupervised(
        self,
        inputs: Sequence[str],
        **kwargs: Any,
    ) -> TrainResult:
        url = urljoin(
            self.config.gradient_api_url,
            f"models/{self.config.model_id}/fine-tune",
        )

        headers = {
            "authorization": f"Bearer {self.config.gradient_access_token}",
            "x-gradient-workspace-id": f"{self.config.gradient_workspace_id}",
            "accept": "application/json",
            "content-type": "application/json",
        }

        payload = {
            "samples": [{"inputs": input} for input in inputs],
            **kwargs,
        }

        response = requests.post(
            url=url,
            headers=headers,
            json=payload,
        )

        if response.status_code != 200:
            raise Exception(
                f"Gradient returned an unexpected response with status "
                f"{response.status}: {response.text}"
            )

        response_json = response.json()
        loss = response_json["sumLoss"] / response_json["numberOfTrainableTokens"]
        return TrainResult(loss=loss)

    async def atrain_unsupervised(
        self,
        inputs: Sequence[str],
        **kwargs: Any,
    ) -> TrainResult:
        url = urljoin(
            self.config.gradient_api_url,
            f"models/{self.config.model_id}/fine-tune",
        )

        headers = {
            "authorization": f"Bearer {self.config.gradient_access_token}",
            "x-gradient-workspace-id": f"{self.config.gradient_workspace_id}",
            "accept": "application/json",
            "content-type": "application/json",
        }

        payload = {
            "samples": [{"inputs": input} for input in inputs],
            **kwargs,
        }

        async with httpx.AsyncClient() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    raise Exception(
                        f"Gradient returned an unexpected response with status "
                        f"{response.status}: {await response.text()}"
                    )

                response_json = await response.json()
                loss = response_json["sumLoss"] / response_json["numberOfTrainableTokens"]
                return TrainResult(loss=loss)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.config.model_kwargs or {}
        return {
            **{"gradient_api_url": self.config.gradient_api_url},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    @final
    def _llm_type(self) -> Literal["gradient"]:
        """Return type of llm."""
        return "gradient"

