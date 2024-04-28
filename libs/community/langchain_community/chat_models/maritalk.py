from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import Field
from requests import Response
from requests.exceptions import HTTPError

class MariTalkHTTPError(HTTPError):
    def __init__(self, request_obj: Response) -> None:
        self.request_obj = request_obj
        self.message = self._parse_response_message(request_obj)
        self.status_code = request_obj.status_code

    def _parse_response_message(self, request_obj: Response) -> str:
        try:
            response_json = request_obj.json()
            if "detail" in response_json:
                api_message = response_json["detail"]
            elif "message" in response_json:
              
