import json
import itertools
from functools import lru_cache
from typing import Any, Iterator, Literal, Mapping, Optional, Sequence, Union

import datasets
from dataclasses import dataclass
from itertools import chain
from langchain_core.documents import Document

@dataclass
class HuggingFaceDatasetLoaderConfig:
    path: str
    page_content_column: str = "text"
    name: Optional[str] = None
    data_dir: Optional[str] = None
    data_files: Optional[
        Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
    ] = None
    cache_dir: Optional[str] = None
    keep_in_memory: Optional[bool] = None
    save_infos: bool = False
    use_auth_token: Optional[Union[bool, str]] = None
    num_proc: Optional[int] = None


class HuggingFaceDatasetLoader(BaseLoader):
    """Load from `Hugging Face Hub` datasets."""

    def __init__(self, config: HuggingFaceDatasetLoaderConfig):
        self.config = config

    @lru_cache(maxsize=None)
    def _parse_obj(self, page_content: Union[str, object]) -> str:
        if isinstance(page_content, object):
            return json.dumps(page_content)
        return page_content

    def lazy_load(self) -> Iterator[Document]:
        """Load documents lazily."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Could not import datasets python package. "
                "Please install it with `pip install datasets`."
            )

        dataset = load_dataset(
            path=self.config.path,
            name=self.config.name,
            data_dir=self.config.data_dir,
            data_files=self.config.data_files,
            cache_dir=self.config.cache_dir,
            keep_in_memory=self.config.keep_in_memory,
            save_infos=self.config.save_infos,
            use_auth_token=self.config.use_auth_token,
            num_proc=self.config.num_proc,
        )

        yield from (
            Document(
                page_content=self._parse_obj(row.pop(self.config.page_content_column)),
                metadata=row,
            )
            for key in dataset.keys()
            for row in dataset[key].map(self._parse_obj, batched=True)
        )
