from __future__ import annotations

from typing import List

from langchain_core.tools import BaseTool
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools.azure_ai_services import (
    AzureAiServicesDocumentIntelligenceTool,
    AzureAiServicesImageAnalysisTool,
    AzureAiServicesSpeechToTextTool,
    AzureAiServicesTextAnalyticsForHealthTool,
    AzureAiServicesTextToSpeechTool,
)

class AzureAiServicesToolkit(BaseToolkit):
    """Toolkit for Azure AI Services."""

    def __init__(self):
        """Initialize the toolkit with a list of Azure AI tools."""
        self.tools = [
            AzureAiServicesDocumentIntelligenceTool(),
            AzureAiServicesImageAnalysisTool(),
            AzureAiServicesSpeechToTextTool(),
            AzureAiServicesTextToSpeechTool(),
            AzureAiServicesTextAnalyticsForHealthTool(),
        ]

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools

