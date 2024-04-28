"""Base interface for loading large language model APIs."""
import json
import pathlib
import sys
import yaml
from typing import Any, Dict, Literal, Type, Union

import typing_extensions
from langchain_core.language_models.llms import BaseLLM
from pydantic import Field

from langchain_community.llms import get_type_to_cls_dict

_ALLOW_DANGEROUS_DESERIALIZATION_ARG = "allow_dangerous_deserialization"


@typing_extensions.final
class LoadLLMFromConfig:
    """Load LLM from Config Dict."""

    def __init__(self, type_to_cls_dict: Dict[str, Type[BaseLLM]]):
        """Initialize LoadLLMFromConfig.

        Args:
            type_to_cls_dict (Dict[str, Type[BaseLLM]]): A dictionary mapping
                LLM type names to their corresponding classes.
        """
        self.type_to_cls_dict = type_to_cls_dict

    def __call__(self, config: Dict[str, Any], **kwargs: Any) -> BaseLLM:
        """Load LLM from Config Dict.

        Args:
            config (Dict[str, Any]): A configuration dictionary.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            BaseLLM: An instance of a large language model.
        """
        if "_type" not in config:
            raise ValueError("Must specify an LLM Type in config")
        config_type = config.pop("_type")

        if config_type not in self.type_to_cls_dict:
            raise ValueError(f"Loading {config_type} LLM not supported")

        llm_cls = self.type_to_cls_dict[config_type]()

        load_kwargs = {}
        if _ALLOW_DANGEROUS_DESERIALIZATION_ARG in llm_cls.__fields__:
            load_kwargs[_ALLOW_DANGEROUS_DESERIALIZATION_ARG] = kwargs.get(
                _ALLOW_DANGEROUS_DESERIALIZATION_ARG, False
            )

        return llm_cls(**config, **load_kwargs)


def load_llm_from_file(
    file: Union[str, pathlib.Path], **kwargs: Any
) -> BaseLLM:
    """Load LLM from a file.

    Args:
        file (Union[str, pathlib.Path]): A file path or name.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        BaseLLM: An instance of a large language model.
    """
    file_path = pathlib.Path(file)

    if file_path.suffix == ".json":
        with open(file_path) as f:
            config = json.load(f)
    elif file_path.suffix.endswith((".yaml", ".yml")):
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("File type must be json or yaml")

    # Load the LLM from the config now.
    return load_llm_from_config(config, **kwargs)


def load_llm(
    file: Union[str, pathlib.Path], **kwargs: Any
) -> BaseLLM:
    """Load LLM from a file.

    Args:
        file (Union[str, pathlib.Path]): A file path or name.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        BaseLLM: An instance of a large language model.
    """
    try:
        return load_llm_from_file(file, **kwargs)
    except Exception:
        sys.excepthook(*sys.exc_info())
        raise

