from pydantic import BaseModel, Extra, Field

from langchain_community.cross_encoders.base import BaseCrossEncoder

class HuggingFaceCrossEncoder(BaseModel, BaseCrossEncoder):
    """HuggingFace cross encoder models.

    Example:
        .. code-block:: python

            from langchain_community.cross_encoders import HuggingFaceCrossEncoder

            model_name = "BAAI/bge-reranker-base"
            model_kwargs = {'device': 'cpu'}
            hf = HuggingFaceCrossEncoder(
                model_name=model_name,
                model_kwargs=model_kwargs
            )
    """

    DEFAULT_MODEL_NAME = "BAAI/bge-reranker-base"

    client: Any
    model_name: str = DEFAULT_MODEL_NAME
    model_kwargs: dict = Field(default_factory=dict)

    def __init__(self, **data: Any):
        super().__init__(**data)
        try:
            import sentence_transformers
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc

        self.client = sentence_transformers.CrossEncoder(
            self.model_name, **self.model_kwargs
        )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute similarity scores using a HuggingFace transformer model.

        Args:
            text_pairs: The list of text text_pairs to score the similarity.

        Returns:
            List of scores, one for each pair.
        """
        scores = self.client.predict(text_pairs)
        return scores * 100  # Normalize scores to be between 0 and 100
