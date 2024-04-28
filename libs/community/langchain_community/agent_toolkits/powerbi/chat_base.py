"""Power BI agent."""
from __future__ import annotations

import typing as ty
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.agent_toolkits.powerbi.prompt import (
    POWERBI_CHAT_PREFIX,
    POWERBI_CHAT_SUFFIX,
)
from langchain_community.agent_toolkits.powerbi.toolkit import PowerBIToolkit
from langchain_community.utilities.powerbi import PowerBIDataset

from langchain.agents import AgentExecutor, ConversationalChatAgent
from langchain.memory import ConversationBufferMemory


def create_pbi_chat_agent(
    llm: BaseChatModel,
    toolkit: Optional[PowerBIToolkit] = None,
    powerbi: Optional[PowerBIDataset] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    output_parser: Optional[Any] = None,
    prefix: str = POWERBI_CHAT_PREFIX,
    suffix: str = POWERBI_CHAT_SUFFIX,
    examples: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    memory: Optional[Any] = None,
    top_k: int = 10,
    verbose: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a Power BI agent from a Chat LLM and tools.

    If you supply only a toolkit and no Power BI dataset, the same LLM is used for both.
    """
    if toolkit is None and powerbi is None:
        raise ValueError("Either a toolkit or powerbi dataset must be provided")

    if toolkit is None:
        toolkit = PowerBIToolkit(powerbi=powerbi, llm=llm, examples=examples)

    tools = toolkit.get_tools()
    tables = powerbi.table_names if powerbi else toolkit.powerbi.table_names

    system_message = prefix.format(top_k=top_k).format(tables=tables)

    agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        system_message=system_message,
        human_message=suffix,
        input_variables=input_variables,
        callback_manager=callback_manager,
        output_parser=output_parser,
        verbose=verbose,
        **kwargs,
    )

    memory = memory or ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        memory=memory,
        verbose=verbose,
    )
