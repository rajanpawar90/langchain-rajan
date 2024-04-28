"""Azure OpenAI embeddings wrapper."""

from __future__ import annotations

import os
import warnings
from typing import Callable, Dict, Optional, Union

from langchain_core._api.deprecation import deprecated
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env

from langchain_community.embeddings.openai import OpenAIEmbeddings


@deprecated(
    since="0.0.9",
    removal="0.2.0",
    alternative_import="langchain_openai.AzureOpenAIEmbeddings",
)
class AzureOpenAIEmbeddings(OpenAIEmbeddings):
    """`Azure OpenAI` Embeddings API."""

    azure_endpoint: Union[str, None] = None
    deployment: Optional[str] = Field(default=None, alias="azure_deployment")
    openai_api_key: Union[str, None] = Field(default=None, alias="api_key")
    azure_ad_token: Union[str, None] = None
    azure_ad_token_provider: Union[Callable[[], str], None] = None
    openai_api_version: Optional[str] = Field(default=None, alias="api_version")
    validate_base_url: bool = True

    @root_validator()
    def validate_environment(self, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["openai_api_key"] = values.get("openai_api_key", os.getenv("AZURE_OPENAI_API_KEY"))
        values["openai_api_base"] = values.get("openai_api_base", os.getenv("OPENAI_API_BASE"))
        values["openai_api_version"] = values.get("openai_api_version", os.getenv("OPENAI_API_VERSION", default="2023-05-15"))  # noqa: E501
        values["openai_organization"] = values.get("openai_organization", os.getenv("OPENAI_ORG_ID", os.getenv("OPENAI_ORGANIZATION")))  # noqa: E501
        values["openai_proxy"] = get_from_dict_or_env(values, "openai_proxy", "OPENAI_PROXY", default="")  # noqa: E501
        values["azure_endpoint"] = values.get("azure_endpoint", os.getenv("AZURE_OPENAI_ENDPOINT"))
        values["azure_ad_token"] = values.get("azure_ad_token", os.getenv("AZURE_OPENAI_AD_TOKEN"))
        values["chunk_size"] = min(values.get("chunk_size", 16), 16)

        try:
            import openai
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        if is_openai_v1():
            openai_api_base = values.get("openai_api_base")
            if openai_api_base and values["validate_base_url"]:
                if "/openai" not in openai_api_base:
                    values["openai_api_base"] += "/openai"
                    warnings.warn(
                        "As of openai>=1.0.0, Azure endpoints should be specified via "
                        f"the `azure_endpoint` param not `openai_api_base` "
                        f"(or alias `base_url`). Updating `openai_api_base` from "
                        f"{openai_api_base} to {values['openai_api_base']}."
                    )
                if values["deployment"]:
                    warnings.warn(
                        "As of openai>=1.0.0, if `deployment` (or alias "
                        "`azure_deployment`) is specified then "
                        "`openai_api_base` (or alias `base_url`) should not be. "
                        "Instead use `deployment` (or alias `azure_deployment`) "
                        "and `azure_endpoint`."
                    )
                    if values["deployment"] not in values["openai_api_base"]:
                        warnings.warn(
                            "As of openai>=1.0.0, if `openai_api_base` "
                            "(or alias `base_url`) is specified it is expected to be "
                            "of the form "
                            "https://example-resource.azure.openai.com/openai/deployments/example-deployment. "  # noqa: E501
                            f"Updating {openai_api_base} to "
                            f"{values['openai_api_base']}."
                        )
                        values["openai_api_base"] += (
                            "/deployments/" + values["deployment"]
                        )
                    values["deployment"] = None
            client_params = {
                "api_version": values["openai_api_version"],
                "azure_endpoint": values["azure_endpoint"],
                "azure_deployment": values["deployment"],
                "api_key": values["openai_api_key"],
                "azure_ad_token": values["azure_ad_token"],
                "azure_ad_token_provider": values["azure_ad_token_provider"],
                "organization": values["openai_organization"],
                "base_url": values["openai_api_base"],
                "timeout": values["request_timeout"],
                "max_retries": values["max_retries"],
                "default_headers": values["default_headers"],
                "default_query": values["default_query"],
                "http_client": values["http_client"],
            }
            values["client"] = openai.AzureOpenAI(**client_params).embeddings  # type: ignore  # noqa: E501
            values["async_client"] = openai.AsyncAzureOpenAI(**client_params).embeddings  # type: ignore  # noqa: E501
        else:
            values["client"] = openai.Embedding

        return values

    @property
    def _llm_type(self) -> str:
        return "azure-openai-chat"
