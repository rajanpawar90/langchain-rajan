from __future__ import annotations

import importlib
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Sequence,
    Union,
    overload,
)

from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.pydantic_v1 import BaseModel
from typing_extensions import Literal


class MessageDict(NamedTuple):
    role: str
    content: str = ""
    function_call: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


