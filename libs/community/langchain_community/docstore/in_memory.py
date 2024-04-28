"""Simple in-memory docstore in the form of a dict."""
from typing import Dict, List, Optional, Union

from langchain_core.documents import Document

from langchain_community.docstore.base import AddableMixin, Docstore


class InMemoryDocstore(Docstore, AddableMixin):
    """Simple in-memory docstore in the form of a dict."""

    def __init__(self, data: Optional[Dict[str, Document]] = None):
        """Initialize with dict."""
        self.data = data or {}

    def add(self, texts: Dict[str, Document]) -> None:
        """Add texts to in-memory dictionary.

        Args:
            texts: dictionary of id -> document.

        Returns:
            None
        """
        overlapping = set(texts).intersection(self.data)
        if overlapping:
            raise ValueError(f"Tried to add ids that already exist: {overlapping}")
        self.data.update(texts)

    def delete(self, ids: List[str]) -> None:
        """Delete IDs from in-memory dictionary."""
        for id_ in ids:
            if id_ not in self.data:
                raise ValueError(f"Tried to delete id {id_} that does not exist.")
        for id_ in ids:
            self.data.pop(id_)

    def search(self, search: Union[str, List[str]]) -> Union[str, Document, List[str]]:
        """Search via direct lookup.

        Args:
            search: id of a document to search for or a list of ids.

        Returns:
            Document if found, error message, or list of messages if search is a list.
        """
        if isinstance(search, str):
            if search not in self.data:
                return f"ID {search} not found."
            return self.data[search]
        elif isinstance(search, list):
            not_found = [f"ID {id_} not found." for id_ in search if id_ not in self.data]
            found = [self.data[id_] for id_ in search if id_ in self.data]
            return not_found + found if not_found else found
        else:
            raise TypeError("Search parameter must be a string or a list of strings.")
