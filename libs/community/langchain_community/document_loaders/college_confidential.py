from typing import List

from bs4 import BeautifulSoup
import requests
from langchain_core.documents import Document

class CollegeConfidentialLoader:
    """Load `College Confidential` webpages."""

    def __init__(self, web_path: str):
        self.web_path = web_path

    def scrape(self) -> BeautifulSoup:
        """Scrape webpage and return BeautifulSoup object."""
        response = requests.get(self.web_path)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup

    def load(self) -> List[Document]:
        """Load webpage as Document."""
        soup = self.scrape()
        text = soup.select_one("main[class='skin-handler']").get_text()
        metadata = {"source": self.web_path}
        return [Document(page_content=text, metadata=metadata)]
