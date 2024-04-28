import re
from pathlib import Path
from typing import Dict, Iterator, Union

from langchain_core.documents import Document

FRONT_MATTER_REGEX = re.compile(r"^---\n(.*?)\n---\n", re.MULTILINE | re.DOTALL)
"""Regex to match front matter metadata in markdown files."""

TASK_REGEX = re.compile(r"\s*-\s\[\s\]\s.*|\s*\[\s\]\s.*")
"""Regex to match tasks in markdown files."""

HASHTAG_REGEX = re.compile(r"#")
"""Regex to match hashtags in markdown files."""

DOCLINK_REGEX = re.compile(r"\[\[.*?\]\]")
"""Regex to match doclinks in markdown files."""


class AcreomLoader(BaseLoader):
    """Load `acreom` vault from a directory."""

    def __init__(
        self,
        path: Union[str, Path],
        encoding: str = "UTF-8",
        collect_metadata: bool = True,
    ):
        """Initialize the loader."""
        self.file_path = Path(path)
        """Path to the directory containing the markdown files."""
        self.encoding = encoding
        """Encoding to use when reading the files."""
        self.collect_metadata = collect_metadata
        """Whether to collect metadata from the front matter."""

    def _parse_front_matter(self, content: str) -> Dict[str, str]:
        """Parse front matter metadata from the content and return it as a dict."""
        front_matter = {}
        for line in content.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                front_matter[key.strip()] = value.strip()
        return front_matter

    def _remove_front_matter(self, content: str) -> str:
        """Remove front matter metadata from the given content."""
        return FRONT_MATTER_REGEX.sub("", content)

    def _process_acreom_content(self, content: str) -> str:
        """Remove acreom specific elements from content that do not contribute to the context of current document."""
        content = TASK_REGEX.sub("", content)
        content = HASHTAG_REGEX.sub("", content)
        content = DOCLINK_REGEX.sub("", content)
        return content

    def lazy_load(self) -> Iterator[Document]:
        """Load markdown files as Documents."""
        for p in self.file_path.glob("**/*.md"):
            try:
                content = p.read_text(encoding=self.encoding)
            except Exception:
                continue

            if self.collect_metadata:
                front_matter = self._parse_front_matter(content)
                content = self._remove_front_matter(content)
            else:
                front_matter = {}

            content = self._process_acreom_content(content)

            metadata = {
                "source": str(p.name),
                "path": str(p),
                **front_matter,
            }

            yield Document(page_content=content, metadata=metadata)
