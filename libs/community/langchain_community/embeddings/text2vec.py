"""Wrapper around text2vec embedding models."""

import typing as tp

import langchain_core as lc_core
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
from text2vec import SentenceModel

