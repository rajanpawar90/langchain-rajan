import logging
from pathlib import Path
from typing import Iterator, Optional, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.helpers import detect_file_encodings

logger = logging.getLogger(__name__)

class TextLoader(BaseLoader):
    """Load text file.

    Args:
        file_path: Path to the file to load.
        encoding: File encoding to use. If `None`, the file will be loaded
            with the default system encoding.
        autodetect_encoding: Whether to try to autodetect the file encoding
            if the specified encoding fails.
        raise_on_exceptions: Whether to raise exceptions or return `None`
            when there is an error loading the file.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,
        raise_on_exceptions: bool = True,
    ):
        """Initialize with file path."""
        self.file_path = file_path
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding
        self.raise_on_exceptions = raise_on_exceptions

    def lazy_load(self) -> Iterator[Document]:
        """Load from file path."""
        return self._load()

    def load(self) -> Optional[Document]:
        """Load from file path."""
        result = self._load()
        if result is not None:
            return next(result)
        return None

    def _load(self) -> Iterator[Document]:
        """Load from file path."""
        text = ""
        try:
            with open(self.file_path, encoding=self.encoding) as f:
                text = f.read()
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(self.file_path)
                for encoding in detected_encodings:
                    logger.debug(f"Trying encoding: {encoding.encoding}")
                    try:
                        with open(self.file_path, encoding=encoding.encoding) as f:
                            text = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    if self.raise_on_exceptions:
                        raise RuntimeError(
                            f"Error loading {self.file_path}: could not "
                            f"autodetect encoding"
                        ) from e
                    else:
                        return
            else:
                if self.raise_on_exceptions:
                    raise RuntimeError(f"Error loading {self.file_path}") from e
                else:
                    return
        except Exception as e:
            if self.raise_on_exceptions:
                raise RuntimeError(f"Error loading {self.file_path}") from e
            else:
                return

        metadata = {"source": str(self.file_path)}
        yield Document(page_content=text, metadata=metadata)

__module__ = "langchain_community.document_loaders.text"
__package__ = "langchain_community.document_loaders"
