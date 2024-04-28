"""JinaChat wrapper."""
from __future__ import annotations

import logging
import sys
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    Literal,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
)

import openai
from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    FieldAliases,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    RootValidator,
    StrictBool,
    StrictDict,
    StrictFloat,
    StrictInt,
    StrictList,
    StrictSet,
    StrictStr,
    StrictTuple,
    ValidatedDict,
    ValidatedOptional,
    ValidatedSet,
    ValidatedStrictDict,
    ValidatedStrictList,
    ValidatedStrictSet,
    ValidatedStrictTuple,
    ValidatedStrictUnion,
    ValidatedStrictType,
    ValidatedTuple,
    ValidatedUnion,
    conint,
    constr,
    nonpositive,
    nonpositiveint,
    past,
    positive,
    positivefloat,
    redefined,
    root_validator,
    none_is_required,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class ChatGenerationChunk(NamedTuple):
    """A single chunk of a chat generation."""

    message: BaseMessageChunk
    """The message chunk."""


class JinaChatConfig(BaseModel):
    """Configuration for JinaChat."""

    temperature: PositiveFloat = Field(
        0.7, title="Temperature", ge=0, le=1, description="Sampling temperature."
    )
    max_retries: PositiveInt = Field(
        6, title="Max Retries", ge=0, description="Maximum number of retries."
    )
    streaming: bool = Field(
        False, title="Streaming", description="Whether to stream the results."
    )
    request_timeout: Union[float, Tuple[float, float]] = Field(
        None,
        title="Request Timeout",
        ge=0,
        description="Timeout for requests to JinaChat completion API.",
    )
    max_tokens: Optional[int] = Field(
        None, title="Max Tokens", ge=None, description="Maximum number of tokens."
    )


class JinaChat(BaseChatModel):
    """`JinaChat` Chat models API.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``JINACHAT_API_KEY`` set to your API key, which you
    can generate at https://chat.jina.ai/api.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly specified.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import JinaChat
            chat = JinaChat()
    """

    jinachat_api_key: ValidatedOptional[StrictStr] = Field(
        default=None, title="API Key", description="JinaChat API Key."
    )
    model_kwargs: ValidatedStrictDict[StrictStr, Any] = Field(
        default_factory=dict,
        title="Model Keywords",
        description="Holds any model parameters valid for `create` call not explicitly specified.",
    )
    _client: Any = PrivateAttr(default=openai.ChatCompletion)
    _config: JinaChatConfig = Field(default_factory=JinaChatConfig)

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["jinachat_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "jinachat_api_key", "JINACHAT_API_KEY")
        )
        try:
            import openai

        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        try:
            values["_client"] = openai.ChatCompletion
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling JinaChat API."""
        return {
            "request_timeout": self._config.request_timeout,
            "max_tokens": self._config.max_tokens,
            "stream": self._config.streaming,
            "temperature": self._config.temperature,
            **self.model_kwargs,
        }

    def _create_retry_decorator(self) -> Callable[[Any], Any]:
        min_seconds = 1
        max_seconds = 60
        # Wait 2^x * 1 second between each retry starting with
        # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
        return retry(
            reraise=True,
            stop=stop_after_attempt(self._config.max_retries),
            wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
            retry=(
                retry_if_exception_type(openai.error.Timeout)
                | retry_if_exception_type(openai.error.APIError)
                | retry_if_exception_type(openai.error.APIConnectionError)
                | retry_if_exception_type(openai.error.RateLimitError)
                | retry_if_exception_type(openai.error.ServiceUnavailableError)
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

    def completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self._client.create(**kwargs)

        return _completion_with_retry(**kwargs)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            for k, v in token_usage.items():
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
        return {"token_usage": overall_token_usage}

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        for chunk in self.completion_with_retry(messages=message_dicts, **params):
            delta = chunk["choices"][0]["delta"]
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk)
            yield cg_chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._config.streaming:
            stream_iter = self._stream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.completion_with_retry(messages=message_dicts, **params)
        return self._create_chat_result(response)

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = dict(self._invocation_params)
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            gen = ChatGeneration(message=message)
            generations.append(gen)
        llm_output = {"token_usage": response["usage"]}
        return ChatResult(generations=generations, llm_output=llm_output)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        async for chunk in await acompletion_with_retry(
            self, messages=message_dicts, **params
        ):
            delta = chunk["choices"][0]["delta"]
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                await run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk)
            yield cg_chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._config.streaming:
            stream_iter = self._astream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = await acompletion_with_retry(self, messages=message_dicts, **params)
        return self._create_chat_result(response)

    @property
    def _invocation_params(self) -> Mapping[str, Any]:
        """Get the parameters used to invoke the model."""
        jinachat_creds: dict = {
            "api_key": self.jinachat_api_key
            and self.jinachat_api_key.get_secret_value(),
            "api_base": "https://api.chat.jina.ai/v1",
            "model": "jinachat",
        }
        return {**jinachat_creds, **self._default_params}

    @property
    def _llm_type(self) -> Literal["jinachat"]:
        """Return type of chat model."""
        return "jinachat"
