import logging
from typing import Iterator, List, Optional

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.query import QueryRow
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

