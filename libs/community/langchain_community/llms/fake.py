import asyncio
import time
from typing import Any, AsyncIterator, Iterator, List, Mapping, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.base import LLM
from langchain_core.runnables import RunnableConfig

class FakeListLLM(LLM):
    """Fake LLM for testing purposes."""

    def __init__(
        self,
        responses: List[str],
        sleep: Optional[float] = None,
    ):
        self.responses = responses
        self.sleep = sleep
        self.i = 0

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake-list"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Return next response"""
        response = self.responses[self.i]
        self.i = (self.i + 1) % len(self.responses)
        return response

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Return next response"""
        response = self.responses[self.i]
        self.i = (self.i + 1) % len(self.responses)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"responses": self.responses}


class FakeStreamingListLLM(FakeListLLM):
    """Fake streaming list LLM for testing purposes."""

    async def astream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,

