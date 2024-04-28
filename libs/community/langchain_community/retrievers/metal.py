from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.exceptions import InvalidClientType
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.retrievers import BaseRetriever
from metal_sdk.metal import Metal

class MetalRetriever(BaseRetriever):
    """`Metal API` retriever."""

    client: Metal
    """The Metal client to use."""
    params: Optional[Dict[str, Any]] = None
    """The parameters to pass to the Metal client."""

    @root_validator(pre=True)
    def validate_client(cls, values: dict) -> dict:
        """Validate that the client is of the correct type."""
        if "client" in values:
            client = values["client"]
            if not isinstance(client, Metal):
                raise InvalidClientType(
                    f"Got unexpected client, should be of type Metal. "
                    f"Instead, got {type(client)}"
                )

        values["params"] = values.get("params", {})

        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.client.search({"text": query}, **self.params)
        final_results = []
        for r in results["data"]:
            metadata = {k: v for k, v in r.items() if k != "text"}
            final_results.append(Document(page_content=r["text"], metadata=metadata))
        return final_results
