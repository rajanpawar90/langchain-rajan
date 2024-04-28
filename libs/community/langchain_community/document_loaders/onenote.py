import os
import re
from typing import Dict, Iterator, List, Optional

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, BaseSettings, Field, FilePath, SecretStr
from msal import ConfidentialClientApplication

class _OneNoteGraphSettings(BaseSettings):
    client_id: str = Field(..., env="MS_GRAPH_CLIENT_ID")
    client_secret: SecretStr = Field(..., env="MS_GRAPH_CLIENT_SECRET")

    class Config:
        env_prefix = ""
        case_sentive = False
        env_file = ".env"

class OneNoteLoader(BaseModel, BaseLoader):
    settings: _OneNoteGraphSettings = Field(default_factory=_OneNoteGraphSettings)
    auth_with_token: bool = False
    notebook_name: Optional[str] = None
    section_name: Optional[str] = None
    page_title: Optional[str] = None
    object_ids: Optional[List[str]] = None
    onenote_api_base_url: str = "https://graph.microsoft.com/v1.0/me/onenote"
    authority_url: str = "https://login.microsoftonline.com/consumers/"
    token_path: FilePath = FilePath(os.path.expanduser("~/.credentials/onenote_graph_token.txt"))
    access_token: str = ""

    def lazy_load(self) -> Iterator[Document]:
        self._auth()
        if self.object_ids is not None:
            for object_id in self.object_ids:
                yield from self._parse_page(self._get_page_content(object_id))
        else:
            for page in self._get_pages():
                yield from self._parse_page(self._get_page_content(page["id"]))

    def _get_page_content(self, page_id: str) -> str:
        request_url = self.onenote_api_base_url + f"/pages/{page_id}/content"
        response = requests.get(request_url, headers=self._headers, timeout=10)
        response.raise_for_status()
        return response.text

    def _parse_page(self, page_content: str) -> Iterator[Document]:
        soup = BeautifulSoup(page_content, "html.parser")
        page_title = ""
        title_tag = soup.title
        if title_tag:
            page_title = title_tag.get_text(strip=True)
        page_content = soup.get_text(separator="\n", strip=True)
        yield Document(page_content=page_content, metadata={"title": page_title})

    @property
    def _headers(self) -> Dict[str, str]:
        if not self._is_token_expired():
            return {"Authorization": f"Bearer {self.access_token}"}
        return {}

    def _auth(self) -> None:
        if self.access_token != "" and not self._is_token_expired():
            return

        if self.auth_with_token:
            if not self.token_path.parent.exists():
                self.token_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with self.token_path.open("r") as token_file:
                    self.access_token = token_file.read().strip()
            except Exception:
                pass
        else:
            client_instance = ConfidentialClientApplication(
                client_id=self.settings.client_id,
                client_credential=self.settings.client_secret.get_secret_value(),
                authority=self.authority_url,
            )

            authorization_request_url = client_instance.get_authorization_request_url(
                self._scopes
            )
            print("Visit the following url to give consent:")  # noqa: T201
            print(authorization_request_url)  # noqa: T201
            authorization_url = input("Paste the authenticated url here:\n")

            authorization_code = re.search(r"code=([^&]*)", authorization_url).group(1)
            access_token_json = client_instance.acquire_token_by_authorization_code(
                code=authorization_code, scopes=self._scopes
            )
            self.access_token = access_token_json["access_token"]

            if not self.token_path.parent.exists():
                self.token_path.parent.mkdir(parents=True, exist_ok=True)

            with self.token_path.open("w") as token_file:
                token_file.write(self.access_token)

    @property
    def _scopes(self) -> List[str]:
        return ["Notes.Read"]

    def _is_token_expired(self) -> bool:
        # Implement your token expiration check here
        return False

    @property
    def _url(self) -> str:
        query_params_list = ["$select=id"]
        filter_list = []
        expand_list = []

        if self.notebook_name is not None:
            filter_list.append(
                f"parentNotebook/displayName eq '{self.notebook_name}'"
            )
            expand_list.append("parentNotebook")
        if self.section_name is not None:
            filter_list.append(
                f"parentSection/displayName eq '{self.section_name}'"
            )
            expand_list.append("parentSection")
        if self.page_title is not None:
            filter_list.append(f"title eq '{self.page_title}'")

        if expand_list:
            query_params_list.append(f"$expand={','.join(expand_list)}")
        if filter_list:
            query_params_list.append(f"$filter={' and '.join(filter_list)}")

        query_params = "&".join(query_params_list)
        if query_params:
            query_params = f"?{query_params}"
        return f"{self.onenote_api_base_url}/pages{query_params}"

    @property
    def _get_pages(self) -> Iterator[Dict[str, str]]:
        request_url = self._url
        while request_url:
            response = requests.get(request_url, headers=self._headers, timeout=10)
            response.raise_for_status()
            pages = response.json()
            for page in pages["value"]:
                yield page
            if "@odata.nextLink" in pages:
                request_url = pages["@odata.nextLink"]
            else:
                request_url = None
