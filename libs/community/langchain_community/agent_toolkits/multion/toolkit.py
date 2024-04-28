"""MultiOn agent toolkit."""
from __future__ import annotations

from typing import List

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.multion.close_session import MultionCloseSession
from langchain_community.tools.multion.create_session import MultionCreateSession
from langchain_community.tools.multion.update_session import MultionUpdateSession


class MultiOnToolkit(BaseToolkit):
    """Toolkit for interacting with the MultiOn API.

    **Security Note**: This toolkit contains tools that interact with the
        user's browser via the MultiOn API which grants an agent
        access to the user's browser.

        Please review the documentation for the MultiOn API to understand
        the security implications of using this toolkit.

        See https://python.langchain.com/docs/security for more information.
    """

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def __init__(self):
        """Initialize the toolkit with tools."""
        self.tools = [
            MultionCreateSession(),
            MultionUpdateSession(),
            MultionCloseSession(),
        ]

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
