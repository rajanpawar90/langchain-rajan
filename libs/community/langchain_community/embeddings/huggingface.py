from typing import Any, Dict, List, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, SecretStr, validator

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_INSTRUCT_MODEL = "hkunlp/instructor-large"
DEFAULT_BGE_MODEL = "BAAI/bge-large-en"
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = (
    "Represent the question for retrieving supporting documents: "
)
DEFAULT_QUERY_BGE_INSTRUCTION_EN = (
    "Represent this question for searching relevant passages: "
)
DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："

VERSION = "0.1.0"


class HuggingFaceEmbeddings(BaseModel, Embeddings):
    # ... (same as original code)


class HuggingFaceInstructEmbeddings(BaseModel, Embeddings):
    # ... (same as original code)


class HuggingFaceBgeEmbeddings(BaseModel, Embeddings):
    # ... (same as original code)

    @validator("api_key", pre=True)
    def validate_api_key(cls, v):
        if not isinstance(v, SecretStr):
            raise ValueError("api_key must be an instance of SecretStr")
        return v

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        response = requests.get(
            self.query_instruction + text,
            headers=self._headers,
            **self.encode_kwargs,
        )
        return response.json()


class HuggingFaceInferenceAPIEmbeddings(BaseModel, Embeddings):
    """Embed texts using the HuggingFace API.

    Requires a HuggingFace Inference API key and a model name.
    """

    api_key: SecretStr
    """Your API key for the HuggingFace Inference API."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    """The name of the model to use for text embeddings."""
    api_url: Optional[str] = None
    """Custom inference endpoint url. None for using default public url."""

    @property
    def _api_url(self) -> str:
        return self.api_url or self._default_api_url

    @property
    def _default_api_url(self) -> str:
        return (
            "https://api-inference.huggingface.co"
            "/models"
            f"/{self.model_name}"
        )

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}

    def __version__(self) -> str:
        return VERSION

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # ... (same as original code)

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        response = requests.get(
            self._api_url,
            headers=self._headers,
            params={"text": text},
        )
        return response.json()[0]["embedding"]
