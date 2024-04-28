import asyncio
import dataclasses
import json
import time
from typing import (
    Any,
    AsyncContextManager,
    Awaitable,
    ClassVar,
    Final,
    List,
    Literal,
    Optional,
    Sequence,
    Self,
    TypeVar,
    Union,
    cast,
)

import astra
from astra.exceptions import AstraConnectionError
from dataclasses import dataclass
from langchain_core._api.deprecation import deprecated
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

DEFAULT_COLLECTION_NAME: Final = "langchain_message_store"

T = TypeVar("T", bound="AstraDBChatMessageHistory")


@dataclass
class _AstraDBCollectionEnvironment:
    collection_name: str
    token: Optional[str] = None
    api_endpoint: Optional[str] = None
    astra_db_client: Optional[astra.Client] = None
    async_astra_db_client: Optional[astra.AsyncClient] = None
    namespace: Optional[str] = None
    setup_mode: Literal["SYNC", "ASYNC", "OFF"] = "SYNC"
    pre_delete_collection: bool = False

    async def aensure_db_setup(self) -> None:
        if self.async_astra_db_client is None:
            if self.token is None or self.api_endpoint is None:
                raise ValueError("token and api_endpoint are required")
            self.async_astra_db_client = astra.async_connect(
                self.api_endpoint,
                self.token,
                namespace=self.namespace,
            )
            if self.pre_delete_collection:
                await self.async_delete_collection()
            await self.async_create_collection()

    async def async_create_collection(self) -> None:
        if self.async_astra_db_client is None:
            raise AstraConnectionError("No connection to Astra DB")
        await self.async_astra_db_client.create_keyspace_and_collection(
            keyspace=self.collection_name,
            collection=self.collection_name,
        )

    async def async_delete_collection(self) -> None:
        if self.async_astra_db_client is None:
            raise AstraConnectionError("No connection to Astra DB")
        await self.async_astra_db_client.delete_keyspace_and_collection(
            keyspace=self.collection_name,
            collection=self.collection_name,
        )

    @property
    def collection(self) -> astra.Collection:
        if self.astra_db_client is None:
            raise AstraConnectionError("No connection to Astra DB")
        return self.astra_db_client.collection(self.collection_name)

    @property
    def async_collection(self) -> astra.AsyncCollection:
        if self.async_astra_db_client is None:
            raise AstraConnectionError("No connection to Astra DB")
        return self.async_astra_db_client.async_collection(self.collection_name)


@deprecated(
    since="0.0.25",
    removal="0.2.0",
    alternative_import="langchain_astradb.AstraDBChatMessageHistory",
)
class AstraDBChatMessageHistory(BaseChatMessageHistory, metaclass=dataclasses.Final):
    collection_name: str
    session_id: str
    astra_env: _AstraDBCollectionEnvironment

    def __init__(
        self,
        *,
        session_id: str,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        astra_db_client: Optional[astra.Client] = None,
        async_astra_db_client: Optional[astra.AsyncClient] = None,
        namespace: Optional[str] = None,
        setup_mode: Literal["SYNC", "ASYNC", "OFF"] = "SYNC",
        pre_delete_collection: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self.session_id = session_id
        self.astra_env = _AstraDBCollectionEnvironment(
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            astra_db_client=astra_db_client,
            async_astra_db_client=async_astra_db_client,
            namespace=namespace,
            setup_mode=setup_mode,
            pre_delete_collection=pre_delete_collection,
        )
        asyncio.run(self.astra_env.aensure_db_setup())

    @property
    def messages(self) -> List[BaseMessage]:
        self.astra_env.ensure_db_setup()
        message_blobs = [
            doc["body_blob"]
            for doc in sorted(
                self.collection.paginated_find(
                    filter={
                        "session_id": self.session_id,
                    },
                    projection={
                        "timestamp": 1,
                        "body_blob": 1,
                    },
                ),
                key=lambda _doc: _doc["timestamp"],
            )
        ]
        items = [json.loads(message_blob) for message_blob in message_blobs]
        messages = messages_from_dict(items)
        return messages

    @messages.setter
    def messages(self, messages: List[BaseMessage]) -> None:
        raise NotImplementedError("Use add_messages instead")

    async def aget_messages(self) -> List[BaseMessage]:
        await self.astra_env.aensure_db_setup()
        docs = self.async_collection.paginated_find(
            filter={
                "session_id": self.session_id,
            },
            projection={
                "timestamp": 1,
                "body_blob": 1,
            },
        )
        sorted_docs = sorted(
            [doc async for doc in docs],
            key=lambda _doc: _doc["timestamp"],
        )
        message_blobs = [doc["body_blob"] for doc in sorted_docs]
        items = [json.loads(message_blob) for message_blob in message_blobs]
        messages = messages_from_dict(items)
        return messages

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        self.astra_env.ensure_db_setup()
        docs = [
            {
                "timestamp": time.time(),
                "session_id": self.session_id,
                "body_blob": json.dumps(message_to_dict(message)),
            }
            for message in messages
        ]
        self.collection.chunked_insert_many(docs)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        await self.astra_env.aensure_db_setup()
        docs = [
            {
                "timestamp": time.time(),
                "session_id": self.session_id,
                "body_blob": json.dumps(message_to_dict(message)),
            }
            for message in messages
        ]
        await self.async_collection.chunked_insert_many(docs)

    def clear(self) -> None:
        self.astra_env.ensure_db_setup()
        self.collection.delete_many(filter={"session_id": self.session_id})

    async def aclear(self) -> None:
        await self.astra_env.aensure_db_setup()
        await self.async_collection.delete_many(filter={"session_id": self.session_id})
