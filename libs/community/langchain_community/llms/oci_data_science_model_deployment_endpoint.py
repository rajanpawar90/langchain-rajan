import logging
from typing import Any, Dict, List, Literal, Optional, TypeVar

import requests
from pydantic_v1 import Field
from typing_extensions import raise_from

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)

