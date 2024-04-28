from __future__ import annotations

import asyncio
import functools
import logging
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests.exceptions import HTTPError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _create_retry_decorator(llm: BaseLLM, return_type: type, reraise_exception: bool = True) -> Callable[[Any], T]:
    min_seconds = 1
    max_seconds = 4
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterward
    return retry(
        reraise=reraise_exception,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(HTTPError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def check_response(resp: Any) -> Any:
    """Check the response from the completion call."""
    if resp.status_code == 200:
        return resp
    elif resp.status_code in [400, 401]:
        raise ValueError(
            f"status_code: {resp.status_code} \n "
            f"code: {resp.code} \n message: {resp.message}"
        )
    else:
        raise HTTPError(
            f"HTTP error occurred: status_code: {resp.status_code} \n "
            f"code: {resp.code} \n message: {resp.message}",
            response=resp,
        )

