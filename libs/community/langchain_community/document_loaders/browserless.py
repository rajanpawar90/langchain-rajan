from typing import Iterator, List, Union

import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

class BrowserlessLoader(BaseLoader):
    """Load webpages with `Browserless` /content endpoint."""

    def __init__(
        self, api_token: str, urls: Union[str, List[str]], text_content: bool = True
    ):
        """Initialize with API token and the URLs to scrape"""
        self.api_token = api_token
        """Browserless API token."""
        self.urls = urls if isinstance(urls, list) else [urls]
        """List of URLs to scrape."""
        self.text_content = text_content

    def _scrape_url(self, url: str) -> dict:
        """Scrape a single URL."""
        response = requests.post(
            "https://chrome.browserless.io/scrape",
            params={
                "token": self.api_token,
            },
            json={
                "url": url,
                "elements": [
                    {
                        "selector": "body",
                    }
                ],
            },
        )
        response.raise_for_status()
        return response.json()

    def _get_content(self, response_data: dict) -> str:
        """Get the content from the response data."""
        return response_data["data"][0]["results"][0]["text"] if self.text_content else response_data["content"]

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load Documents from URLs."""
        for url in self.urls:
            response_data = self._scrape_url(url)
            content = self._get_content(response_data)
            yield Document(
                page_content=content,
                metadata={
                    "source": url,
                },
            )
