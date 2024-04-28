from typing import Any, Iterator

from langchain_core.documents import Document
from baidubce.services.bos.bos_client import BosClient

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.baiducloud_bos_file import BaiduBOSFileLoader

class BosClientWrapper:
    """A wrapper class for Baidu BosClient to handle imports."""

    def __init__(self, conf: Any):
        self.conf = conf

    def __call__(self, *args, **kwargs):
        return BosClient(self.conf, *args, **kwargs)

