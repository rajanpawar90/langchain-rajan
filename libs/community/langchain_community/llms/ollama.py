import json
import typing
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Union

import aiohttp
import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import GenerationChunk, LLMResult

class OllamaEndpointNotFoundError(Exception):
    """Raised when the Ollama endpoint is not found."""

class _OllamaCommon(BaseLanguageModel):
    base_url: str = "http://localhost:11434"
    model: str = "llama2"
    mirostat: Optional[int] = None
    mirostat_eta: Optional[float] = None
    mirostat_tau: Optional[float] = None
    num_ctx: Optional[int] = None
    num_gpu: Optional[int] = None
    num_thread: Optional[int] = None
    num_predict: Optional[int] = None
    repeat_last_n: Optional[int] = None
    repeat_penalty: Optional[float] = None
    temperature: Optional[float] = None
    stop: Optional[List[str]] = None
    tfs_z: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    system: Optional[str] = None
    template: Optional[str] = None
    format: Optional[str] = None
    timeout: Optional[int] = None
    keep_alive: Optional[Union[int, str]] = None
    headers: Optional[dict] = None

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "format": self.format,
            "options": {
                "mirostat": self.mirostat,
                "mirostat_eta": self.mirostat_eta,
                "mirostat_tau": self.mirostat_tau,
                "num_ctx": self.num_ctx,
                "num_gpu": self.num_gpu,
                "num_thread": self.num_thread,
                "num_predict": self.num_predict,
                "repeat_last_n": self.repeat_last_n,
              
