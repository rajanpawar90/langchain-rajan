import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import requests
from lxml import etree
from pydantic import BaseModel, root_validator

class DocugamiLoader(BaseModel):
    api: str = "https://api.docugami.com/v1preview1"
    access_token: Optional[str] = os.environ.get("DOCUGAMI_API_KEY")
    max_text_length: int = 4096
    min_text_length: int = 32
    max_metadata_length: int = 512
    include_xml_tags: bool = False
    parent_hierarchy_levels: int = 0
    parent_id_key: str = "doc_id"
    sub_chunk_tables: bool = False
    whitespace_normalize_text: bool = True
    docset_id: Optional[str]
    document_ids: Optional[Sequence[str]]
    file_paths: Optional[Sequence[Union[Path, str]]]
    include_project_metadata_in_doc_metadata: bool = True

    @classmethod
    def __init_subclass__(cls, **kwargs):

