from typing import Any, Callable, Dict, Iterator, List, Optional, Type, Union

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str
from langchain_core.utils.env import get_from_dict_or_env
from langchain_community.adapters.openai import convert_message_to_dict

import fireworks.client

def _convert_delta_to_message_chunk(
    _dict: Any, default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    """Convert a delta response to a message chunk."""
    role = _dict.role
    content = _dict.content or ""
    additional_kwargs: Dict[str, Any] = {}

    if role == "user":
        return HumanMessageChunk(content=content)
    elif role == "assistant":
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessageChunk(content=content)
    elif role == "function":
        return FunctionMessageChunk(content=content, name=_dict.name)
    elif role == "chat":
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)


def convert_dict_to_message(_dict: Any) -> BaseMessage:
    """Convert a dict response to a message."""
    role = _dict.role
    content = _dict.content or ""
    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        content = _dict.content
        additional_kwargs: Dict[str, Any] = {}
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=content)
    elif role == "function":
        return FunctionMessage(content=content, name=_dict.name)
    else:
        return ChatMessage(content=content, role=role)


@deprecated(
    since="0.0.26",
    removal="0.2",
    alternative_import="langchain_fireworks.ChatFireworks",
)
class ChatFireworks(BaseChatModel):
    """Fireworks Chat models."""

    model: str = Field(default="accounts/fireworks/models/llama-v2-7b-chat")
    model_kwargs: dict = Field(default_factory=dict)
    fireworks_api_key: Optional[SecretStr] = Field(default=None)
    max_retries: int = 20
    use_retry: bool = True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"fireworks_api_key": "FIREWORKS_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "fireworks"]

    @root_validator()
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that api key in environment."""
        fireworks_api_key = convert_to_secret_str(
            get_from_dict_or_env(values, "fireworks_api_key", "FIREWORKS_API_KEY")
        )
        fireworks.client.api_key = fireworks_api_key.get_secret_value()
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fireworks-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = self._create_message_dicts(messages)

        params = {
            "model": self.model,
            "messages": message_dicts,
            **self.model_kwargs,
            **kwargs,
        }
        response = completion_with_retry(
            self,
            self.use_retry,
            run_manager=run_manager,
            stop=stop,
            **params,
        )
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = self._create_message_dicts(messages)
        params = {
            "model": self.model,
            "messages": message_dicts,
            **self.model_kwargs,
            **kwargs,
        }
        response = await acompletion_with_retry(
            self, self.use_retry, run_manager=run_manager, stop=stop, **params
        )
        return self._create_chat_result(response)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        return llm_outputs[0] if llm_outputs else {}

    def _create_chat_result(self, response: Any) -> ChatResult:
        generations = [
            ChatGeneration(
                message=convert_dict_to_message(res.message),
                generation_info=dict(finish_reason=res.finish_reason),
            )
            for res in response.choices
        ]
        return ChatResult(generations=generations, llm_output=dict(model=self.model))

    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        return [convert_message_to_dict(m) for m in messages]

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts = self._create_message_dicts(messages)

        params = {
            "model": self.model,
            "messages": message_dicts,
            "stream": True,
            **self.model_kwargs,
            **kwargs,
        }
        for chunk in completion_with_retry(
            self, self.use_retry, run_manager=run_manager, stop=stop, **params
        ):
            choice = chunk.choices[0]
            chunk = _convert_delta_to_message_chunk(choice.delta, AIMessageChunk)
            finish_reason = choice.finish_reason
            generation_info = dict(finish_reason=finish_reason) if finish_reason else None
            yield ChatGenerationChunk(
                message=chunk, generation_info=generation_info
            )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts = self._create_message_dicts(messages)

        params = {
            "model": self.model,
            "messages": message_dicts,
            "stream": True,
            **self.model_kwargs,
            **kwargs,
        }
        async for chunk in await acompletion_with_retry_streaming(
            self, self.use_retry, run_manager=run_manager, stop=stop, **params
        ):
            choice = chunk.choices[0]
            chunk = _convert_delta_to_message_chunk(choice.delta, AIMessageChunk)
            finish_reason = choice.finish_reason
            generation_info = dict(finish_reason=finish_reason) if finish_reason else None
            yield ChatGenerationChunk(
                message=chunk, generation_info=generation_info
            )


def conditional_decorator(
    condition: bool, decorator: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    """Define conditional decorator.

    Args:
        condition: The condition.
        decorator: The decorator.

    Returns:
        The decorated function.
    """

    def actual_decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if condition:
            return decorator(func)
        return func

    return actual_decorator


def completion_with_retry(
    llm: ChatFireworks,
    use_retry: bool,
    *,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    errors = (
        fireworks.client.error.RateLimitError,
        fireworks.client.error.InternalServerError,
        fireworks.client.error.BadGatewayError,
        fireworks.client.error.ServiceUnavailableError,
    )
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )(completion_call)(llm, use_retry, run_manager=run_manager, **kwargs)


async def acompletion_with_retry(
    llm: ChatFireworks,
    use_retry: bool,
    *,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call."""
    errors = (
        fireworks.client.error.RateLimitError,
        fireworks.client.error.InternalServerError,
        fireworks.client.error.BadGatewayError,
        fireworks.client.error.ServiceUnavailableError,
    )
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )(acompletion_call)(llm, use_retry, run_manager=run_manager, **kwargs)


async def acompletion_with_retry_streaming(
    llm: ChatFireworks,
    use_retry: bool,
    *,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call for streaming."""
    errors = (
        fireworks.client.error.RateLimitError,
        fireworks.client.error.InternalServerError,
        fireworks.client.error.BadGatewayError,
        fireworks.client.error.ServiceUnavailableError,
    )
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )(acompletion_call)(llm, use_retry, run_manager=run_manager, **kwargs)


def completion_call(
    llm: ChatFireworks,
    use_retry: bool,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Make the completion call."""
    return fireworks.client.ChatCompletion.create(**kwargs)


create_base_retry_decorator = create_base_retry_decorator  # type: ignore
