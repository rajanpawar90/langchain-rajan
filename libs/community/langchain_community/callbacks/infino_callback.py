import time
import importlib
from typing import Any, Dict, List, Literal, Optional, TypeVar

import logging
from logging import Logger
from typing_extensions import LiteralString
from dataclasses import dataclass
from enum import Enum
import pathlib
import contextlib
import tiktoken
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

T = TypeVar("T")


class LogLevel(Enum):
    INFO = "info"
    DEBUG = "debug"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class InfinoConfig:
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    verbose: bool = False


def import_module(module_name: str, error_msg: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        logging.error(error_msg)
        raise e from e


def import_tiktoken() -> tiktoken:
    return import_module("tiktoken", "To use the ChatOpenAI model with Infino callback manager, you need to have the `tiktoken` python package installed.")


def get_num_tokens(string: str, openai_model_name: LiteralString) -> int:
    encoding = tiktoken.encoding_for_model(openai_model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class InfinoCallbackHandler(BaseCallbackHandler):
    def __init__(self, config: InfinoConfig) -> None:
        self.config = config
        self.client = import_infino()
        self.logger: Logger = logging.getLogger(__name__)
        self.is_chat_openai_model = False
        self.chat_openai_model_name = "gpt-3.5-turbo"

    def _send_to_infino(
        self,
        key: str,
        value: Any,
        is_ts: bool = True,
    ) -> None:
        """Send the key-value to Infino.

        Parameters:
        key (str): the key to send to Infino.
        value (Any): the value to send to Infino.
        is_ts (bool): if True, the value is part of a time series, else it
                      is sent as a log message.
        """
        payload = {
            "date": int(time.time()),
            key: value,
            "labels": {
                "model_id": self.config.model_id,
                "model_version": self.config.model_version,
            },
        }

        if self.config.verbose:
            self.logger.info(f"Tracking {key} with Infino: {payload}")

        if is_ts:
            self.client.append_ts(payload)
        else:
            self.client.append_log(payload)

    # ... rest of the class methods
