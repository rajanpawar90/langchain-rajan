import os
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class SearchDepth(Enum):
    """Search depth as enumerator."""

    BASIC = "basic"
    ADVANCED = "advanced"

class TavilySearchAPIRetriever(BaseRetriever):
    """Tavily Search API retriever."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.k = 10
        self.include_generated_answer = False
        self.include_raw_content = False
        self.include_images = False
        self.search_depth = SearchDepth.BASIC
        self.include_domains = None
        self.exclude_domains = None
        self.kwargs = {}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            import tavily
        except ImportError:
            raise ImportError(
                "Tavily python package not found. "
              
