import importlib
import logging
from typing import Any, Callable, List, Optional

from langchain_community.embeddings.self_hosted import SelfHostedEmbeddings

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_INSTRUCT_MODEL = "hkunlp/instructor-large"
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = (
    "Represent the question for retrieving supporting documents: "
)

logger = logging.getLogger(__name__)


def _embed_documents(client: Any, texts: List[str]) -> List[List[float]]:
    """Inference function to send to the remote hardware.

    Accepts a sentence_transformer model_id and
    returns a list of embeddings for each document in the batch.
    """
    instruction_pairs = [(DEFAULT_EMBED_INSTRUCTION, text) for text in texts]
    embeddings = client(instruction_pairs)
    return [embedding.tolist() for embedding in embeddings]


def load_embedding_model(model_id: str, instruct: bool = False, device: int = 0) -> Any:
    """Load the embedding model."""
    try:
        if not instruct:
            import sentence_transformers

            client = sentence_transformers.SentenceTransformer(model_id)
        else:
            if importlib.util.find_spec("InstructorEmbedding") is None:
                raise ImportError("InstructorEmbedding not found")
            from InstructorEmbedding import INSTRUCTOR

            client = INSTRUCTOR(model_id)

        if importlib.util.find_spec("torch") is not None:
            import torch

            cuda_device_count = torch.cuda.device_count()
            if device < -1 or (device >= cuda_device_count):
                raise ValueError(
                    f"Got device=={device}, "
                    f"device is required to be within [-1, {cuda_device_count})"
                )
            if device < 0 and cuda_device_count > 0:
                logger.warning(
                    "Device has %d GPUs available. "
                    "Provide device={deviceId} to `from_model_id` to use available"
                    "GPUs for execution. deviceId is -1 for CPU and "
                    "can be a positive integer associated with CUDA device id.",
                    cuda_device_count,
                )

        client = client.to(device)
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        raise

    return client


class SelfHostedHuggingFaceEmbeddings(SelfHostedEmbeddings):
    """HuggingFace embedding models on self-hosted remote hardware.

    Supported hardware includes auto-launched instances on AWS, GCP, Azure,
    and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another cloud
    like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import SelfHostedHuggingFaceEmbeddings
            import runhouse as rh
            model_id = "sentence-transformers/all-mpnet-base-v2"
            gpu = rh.cluster(name="rh-a10x", instance_type="A100:1")
            hf = SelfHostedHuggingFaceEmbeddings(model_id=model_id, hardware=gpu)
    """

    client: Any  #: :meta private:
    model_id: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    model_reqs: List[str] = ["./", "sentence_transformers", "torch"]
    """Requirements to install on hardware to inference the model."""
    hardware: Any
    """Remote hardware to send the inference function to."""
    model_load_fn: Callable = load_embedding_model
    """Function to load the model remotely on the server."""
    load_fn_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model load function."""
    inference_fn: Callable = _embed_documents
    """Inference function to extract the embeddings."""

    def __init__(self, **kwargs: Any):
        """Initialize the remote inference function."""
        load_fn_kwargs = kwargs.pop("load_fn_kwargs", {})
        load_fn_kwargs["model_id"] = load_fn_kwargs.get("model_id", DEFAULT_MODEL_NAME)
        load_fn_kwargs["instruct"] = load_fn_kwargs.get("instruct", False)
        load_fn_kwargs["device"] = load_fn_kwargs.get("device", 0)
        super().__init__(load_fn_kwargs=load_fn_kwargs, **kwargs)

    def __str__(self) -> str:
        return (
            f"SelfHostedHuggingFaceEmbeddings(model_id={self.model_id},"
            f"hardware={self.hardware},"
            f"model_load_fn={self.model_load_fn},"
            f"load_fn_kwargs={self.load_fn_kwargs},"
            f"inference_fn={self.inference_fn})"
        )

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace instruct model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        instruction_pair = [DEFAULT_QUERY_INSTRUCTION, text]
        embedding = self.client(self.pipeline_ref, [instruction_pair])[0]
        return embedding.tolist()


class SelfHostedHuggingFaceInstructEmbeddings(SelfHostedHuggingFaceEmbeddings):
    """HuggingFace InstructEmbedding models on self-hosted remote hardware.

    Supported hardware includes auto-launched instances on AWS, GCP, Azure,
    and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another
    cloud like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import SelfHostedHuggingFaceInstructEmbeddings
            import runhouse as rh
            model_name = "hkunlp/instructor-large"
            gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')
            hf = SelfHostedHuggingFaceInstructEmbeddings(
                model_name=model_name, hardware=gpu)
    """  # noqa: E501

    model_id: str = DEFAULT_INSTRUCT_MODEL
    """Model name to use."""
    embed_instruction: str = DEFAULT_EMBED_INSTRUCTION
    """Instruction to use for embedding documents."""
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    """Instruction to use for embedding query."""
    model_reqs: List[str] = ["./", "InstructorEmbedding", "torch"]
    """Requirements to install on hardware to inference the model."""

    def __init__(self, **kwargs: Any):
        """Initialize the remote inference function."""
        load_fn_kwargs = kwargs.pop("load_fn_kwargs", {})
        load_fn_kwargs["model_id"] = load_fn_kwargs.get(
            "model_id", DEFAULT_INSTRUCT_MODEL
        )
        load_fn_kwargs["instruct"] = load_fn_kwargs.get("instruct", True)
        load_fn_kwargs["device"] = load_fn_kwargs.get("device", 0)
        super().__init__(load_fn_kwargs=load_fn_kwargs, **kwargs)
        self.model_name = self.model_id

    def __str__(self) -> str:
        return (
            f"SelfHostedHuggingFaceInstructEmbeddings(model_id={self.model_id},"
            f"hardware={self.hardware},"
            f"model_load_fn={self.model_load_fn},"
            f"load_fn_kwargs={self.load_fn_kwargs},"
            f"inference_fn={self.inference_fn})"
        )
