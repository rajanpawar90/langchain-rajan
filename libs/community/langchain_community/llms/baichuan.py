from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

from langchain_community.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)

class BaichuanLLM(LLM):
    # TODO: Adding streaming support.
    """Wrapper around Baichuan large language models."""

    model: str = "Baichuan2-Turbo-192k"
    """
    Other models are available at https://platform.baichuan-ai.com/docs/api.
    """
    temperature: float = 0.3
    top_p: float = 0.95
    timeout: int = 60
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    baichuan_api_host: Optional[str] = None
    baichuan_api_key: Optional[SecretStr] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        api_key = convert_to_secret_str(
            get_from_dict_or_env(values, "baichuan_api_key", "BAICHUAN_API_KEY")
        )
        values["baichuan_api_key"] = api_key
        values["baichuan_api_host"] = get_from_dict_or_env(
            values,
            "baichuan_api_host",
            "BAICHUAN_API_HOST",
            default="https://api.baichuan-ai.com/v1/chat/completions",
        )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            **self.model_kwargs,
        }

    def _post(self, request: Any) -> Any:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.baichuan_api_key.get_secret_value()}",  # type: ignore[union-attr]
        }
        try:
            response = requests.post(
                self.baichuan_api_host,  # type: ignore[arg-type]
              
