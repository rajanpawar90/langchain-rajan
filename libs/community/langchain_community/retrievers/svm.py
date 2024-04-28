from __future__ import annotations

import concurrent.futures
import numpy as np
import sklearn
from typing import Any, Iterable, List, Optional

import langchain_core.callbacks
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever


class SVMRetriever(BaseRetriever):
    """`SVM` retriever.

    Largely based on
    https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb
    """

    embeddings: Embeddings
    """Embeddings model to use."""
    index: np.ndarray
    """Index of embeddings."""
    texts: List[str]
    """List of texts to index."""
    metadatas: Optional[List[dict]] = None
    """List of metadatas corresponding with each text."""
    k: int = 4
    """Number of results to return."""
    relevancy_threshold: Optional[float] = None
    """Threshold for relevancy."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __init__(
        self,
        embeddings: Embeddings,
        index: np.ndarray,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
    ):
        self.embeddings = embeddings
        self.index = index
        self.texts = texts
        self.metadatas = metadatas

    @staticmethod
    def _embed_texts(texts: List[str], embeddings: Embeddings) -> np.ndarray:
        """Embed a list of texts using the given embeddings.

        Args:
            texts: List of texts to embed.
            embeddings: Embeddings model to use.

        Returns:
            Embedded texts as a 2D numpy array.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            raw_embeds = list(executor.map(embeddings.embed_query, texts))
        return np.array(raw_embeds)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embeddings: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> SVMRetriever:
        """Create an SVMRetriever instance from a list of texts.

        Args:
            texts: List of texts to index.
            embeddings: Embeddings model to use.
            metadatas: Optional list of metadatas corresponding with each text.
            **kwargs: Additional keyword arguments.

        Returns:
            An SVMRetriever instance.
        """
        if sklearn is None:
            raise ImportError(
                "Could not import scikit-learn, please install with `pip install "
                "scikit-learn`."
            )

        if numpy is None:
            raise ImportError(
                "Could not import numpy, please install with `pip install numpy`."
            )

        index = cls._embed_texts(texts, embeddings)
        return cls(embeddings=embeddings, index=index, texts=texts, metadatas=metadatas)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        embeddings: Embeddings,
        **kwargs: Any,
    ) -> SVMRetriever:
        """Create an SVMRetriever instance from a list of Document objects.

        Args:
            documents: List of Document objects to index.
            embeddings: Embeddings model to use.
            **kwargs: Additional keyword arguments.

        Returns:
            An SVMRetriever instance.
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        if len(texts) != len(metadatas):
            raise ValueError("Length of texts and metadatas must be equal.")

        return cls.from_texts(texts=texts, embeddings=embeddings, metadatas=metadatas, **kwargs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: langchain_core.callbacks.CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get relevant documents for a given query.

        Args:
            query: The query to search for.
            run_manager: The callback manager for the retriever run.

        Returns:
            List of relevant documents.
        """
        try:
            from sklearn import svm
        except ImportError:
            raise ImportError(
                "Could not import scikit-learn, please install with `pip install "
                "scikit-learn`."
            )

        if not self.texts:
            raise ValueError("No texts have been indexed.")

        query_embeds = np.array(self.embeddings.embed_query(query))
        x = np.concatenate([query_embeds[None, ...], self.index])
        y = np.zeros(x.shape[0])
        y[0] = 1

        clf = svm.LinearSVC(
            class_weight="balanced", verbose=False, max_iter=10000, tol=1e-6, C=0.1
        )
        clf.fit(x, y)

        similarities = clf.decision_function(x)
        sorted_ix = np.argsort(-similarities)

        # svm.LinearSVC in scikit-learn is non-deterministic.
        # if a text is the same as a query, there is no guarantee
        # the query will be in the first index.
        # this performs a simple swap, this works because anything
        # left of the 0 should be equivalent.
        zero_index = np.where(sorted_ix == 0)[0][0]
        if zero_index != 0:
            sorted_ix[0], sorted_ix[zero_index] = sorted_ix[zero_index], sorted_ix[0]

        denominator = np.max(similarities) - np.min(similarities) + 1e-6
        normalized_similarities = (similarities - np.min(similarities)) / denominator

        top_k_results = []
        for row in sorted_ix[1 : self.k + 1]:
            if (
                self.relevancy_threshold is None
                or normalized_similarities[row] >= self.relevancy_threshold
            ):
                metadata = self.metadatas[row - 1] if self.metadatas else {}
                doc = Document(page_content=self.texts[row - 1], metadata=metadata)
                top_k_results.append(doc)
        return top_k_results
