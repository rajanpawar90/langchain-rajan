"""Test ChatModel chat model."""
import asyncio
from typing import Any, AsyncIterator, Awaitable, List, TypeVar

from __module_name__.chat_models import ChatModel

T = TypeVar("T")


async def async_assert(coroutine: Awaitable[T]) -> T:
    """Assert and return the result of an asynchronous operation.

    Args:
        coroutine (Awaitable[T]): The asynchronous operation to execute.

    Returns:
        T: The result of the asynchronous operation.
    """
    result = await coroutine
    assert result is not None
    return result


async def test_astream() -> None:
    """Test streaming tokens from ChatModel."""
    llm = ChatModel()
    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_abatch() -> None:
    """Test batch tokens from ChatModel."""
    llm = ChatModel()
    result: List[Any] = await async_assert(llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"]))
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatModel."""
    llm = ChatModel()
    result = await async_assert(llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]}))
    assert isinstance(result, str)


def test_stream() -> None:
    """Test streaming tokens from ChatModel."""
    llm = ChatModel()
    loop = asyncio.get_event_loop()
    tokens: AsyncIterator = loop.run_until_complete(llm.stream("I'm Pickle Rick"))

    for token in tokens:
        assert isinstance(token.content, str)


def test_batch() -> None:
    """Test batch tokens from ChatModel."""
    llm = ChatModel()
    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


def test_invoke() -> None:
    """Test invoke tokens from ChatModel."""
    llm = ChatModel()
    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
