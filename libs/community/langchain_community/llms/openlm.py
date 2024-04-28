from typing import Any, Dict

import openlm
from pydantic import root_validator

from langchain_core.pydantic_v1 import BaseModel
from langchain_community.llms.openai import BaseOpenAI

class OpenLM(BaseOpenAI, BaseModel):
    """OpenLM models."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{"model": self.model_name}, **super()._invocation_params}

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        if not values.get("client", False):
            try:
                values["client"] = openlm.Completion
            except ImportError:
                raise ImportError(
                    "Could not import openlm python package. "
                    "Please install it with `pip install openlm`."
                )
        if values.get("streaming", False):
            raise ValueError("Streaming not supported with openlm")
        return values
