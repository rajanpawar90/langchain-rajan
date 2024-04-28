import functools
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union

import yaml
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class ObsidianLoader(BaseLoader):
    """Load `Obsidian` files from directory.

    This class is responsible for loading Obsidian files from a directory and
    converting them into Langchain documents.
    """

    FRONT_MATTER_REGEX = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
    TEMPLATE_VARIABLE_REGEX = re.compile(r"{{\s*(.*?)\s*}}", re.DOTALL)
    TAG_REGEX = re.compile(r"[^\S\/]#([a-zA-Z_]+[-_/\w]*)")
    DATAVIEW_LINE_REGEX = re.compile(r"^\s*(\w+)::\s*(.*)$", re.MULTILINE)
    DATAVIEW_INLINE_BRACKET_REGEX = re.compile(r"\[(\w+)::\s*(.*)\]")
    DATAVIEW_INLINE_PAREN_REGEX = re.compile(r"\((\w+)::\s*(.*)\)")

    def __init__(
        self,
        path: Union[str, Path],
        encoding: str = "UTF-8",
        collect_metadata: bool = True,
    ):
        """Initialize with a path.

        Args:
            path: Path to the directory containing the Obsidian files.
            encoding: Charset encoding, defaults to "UTF-8"
            collect_metadata: Whether to collect metadata from the front matter.
                Defaults to True.
        """
        self.file_path = Path(path)
        self.encoding = encoding
        self.collect_metadata = collect_metadata

    def _replace_template_var(
        self, placeholders: Dict[str, str], match: re.Match
    ) -> str:
        """Replace a template variable with a placeholder.

        Args:
            placeholders: A dictionary to store the placeholders and their
                corresponding values.
            match: A regex match object containing the template variable.

        Returns:
            The placeholder string.
        """
        placeholder = f"__TEMPLATE_VAR_{len(placeholders)}__"
        placeholders[placeholder] = match.group(1)
        return placeholder

    def _restore_template_vars(self, obj: Any, placeholders: Dict[str, str]) -> Any:
        """Restore template variables replaced with placeholders to original values.

        Args:
            obj: The object containing the placeholders.
            placeholders: A dictionary containing the placeholders and their
                corresponding values.

        Returns:
            The object with restored template variables.
        """
        if isinstance(obj, str):
            for placeholder, value in placeholders.items():
                obj = obj.replace(placeholder, f"{{{{{value}}}}}")
        elif isinstance(obj, Mapping):
            for key, value in obj.items():
                obj[key] = self._restore_template_vars(value, placeholders)
        elif isinstance(obj, List):
            for i, item in enumerate(obj):
                obj[i] = self._restore_template_vars(item, placeholders)
        return obj

    def _parse_front_matter(self, content: str) -> Optional[Mapping[str, Any]]:
        """Parse front matter metadata from the content and return it as a dict.

        Args:
            content: The content of the Obsidian file.

        Returns:
            A dictionary containing the front matter metadata or None if no
            front matter is found.
        """
        if not self.collect_metadata:
            return None

        match = self.FRONT_MATTER_REGEX.search(content)
        if not match:
            return None

        placeholders: Dict[str, str] = {}
        replace_template_var = functools.partial(
            self._replace_template_var, placeholders
        )
        front_matter_text = self.TEMPLATE_VARIABLE_REGEX.sub(
            replace_template_var, match.group(1)
        )

        try:
            front_matter = yaml.safe_load(front_matter_text)
            return self._restore_template_vars(front_matter, placeholders)
        except yaml.parser.ParserError:
            logger.warning("Encountered non-yaml frontmatter")
            return None

    def _to_langchain_compatible_metadata(self, metadata: Mapping[str, Any]) -> dict:
        """Convert a dictionary to a compatible with langchain.

        Args:
            metadata: A dictionary containing metadata.

        Returns:
            A dictionary with all values converted to strings.
        """
        result = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float)):
                result[key] = value
            else:
                result[key] = str(value)
        return result

    def _parse_document_tags(self, content: str) -> List[str]:
        """Return a list of all tags in within the document.

        Args:
            content: The content of the Obsidian file.

        Returns:
            A list of tags found in the document.
        """
        if not self.collect_metadata:
            return []

        return self.TAG_REGEX.findall(content)

    def _parse_dataview_fields(self, content: str) -> Mapping[str, str]:
        """Parse obsidian dataview plugin fields from the content and return it
        as a dict.

        Args:
            content: The content of the Obsidian file.

        Returns:
            A dictionary containing the dataview fields.
        """
        if not self.collect_metadata:
            return {}

        return {
            **{
                match[0]: match[1]
                for match in self.DATAVIEW_LINE_REGEX.findall(content)
            },
            **{
                match[0]: match[1]
                for match in self.DATAVIEW_INLINE_PAREN_REGEX.findall(content)
            },
            **{
                match[0]: match[1]
                for match in self.DATAVIEW_INLINE_BRACKET_REGEX.findall(content)
            },
        }

    def _remove_front_matter(self, content: str) -> str:
        """Remove front matter metadata from the given content.

        Args:
            content: The content of the Obsidian file.

        Returns:
            The content without front matter metadata.
        """
        if not self.collect_metadata:
            return content
        return self.FRONT_MATTER_REGEX.sub("", content)

    def lazy_load(self) -> Iterator[Document]:
        """Load Obsidian files and convert them into Langchain documents.

        Yields:
            Langchain documents.
        """
        try:
            paths = list(self.file_path.glob("**/*.md"))
        except FileNotFoundError:
            return

        for path in paths:
            try:
                with open(path, encoding=self.encoding) as f:
                    text = f.read()
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(f"Error reading file {path}: {e}")
                continue

            front_matter = self._parse_front_matter(text)
            tags = self._parse_document_tags(text)
            dataview_fields = self._parse_dataview_fields(text)
            text = self._remove_front_matter(text)
            metadata = {
                "source": str(path.name),
                "path": str(path),
                "created": path.stat().st_ctime,
                "last_modified": path.stat().st_mtime,
                "last_accessed": path.stat().st_atime,
                **self._to_langchain_compatible_metadata(front_matter or {}),
                **dataview_fields,
            }

            if tags or (front_matter and "tags" in front_matter):
                metadata["tags"] = ",".join(set(tags + (front_matter.get("tags") or [])))

            yield Document(page_content=text, metadata=metadata)
