"""Cassandra-based chat message history, based on cassIO."""
from __future__ import annotations

import json
import typing
from typing import List, Optional

import uuid
from cassandra.cluster import Session
from cassio.table import ClusteredCassandraTable

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

DEFAULT_TABLE_NAME = "message_store"
DEFAULT_TTL_SECONDS = None

class CassandraChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in Cassandra.

    Args:
        session_id: arbitrary key that is used to store the messages
            of a single chat session.
        session: a Cassandra `Session` object (an open DB connection)
        keyspace: name of the keyspace to use.
        table_name: name of the table to use.
        ttl_seconds: time-to-live (seconds) for automatic expiration
            of stored entries. None (default) for no expiration.
    """

    def __init__(
        self,
        session_id: str,
        session: Session,
        keyspace: str,
        table_name: str = DEFAULT_TABLE_NAME,
        ttl_seconds: Optional[int] = DEFAULT_TTL_SECONDS,
    ) -> None:
        try:
            self.table = ClusteredCassandraTable(
                session=session,
                keyspace=keyspace,
                table=table_name,
                ttl_seconds=ttl_seconds,
                primary_key_type=["TEXT", "TIMEUUID"],
                ordering_in_partition="DESC",
            )
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import cassio python package. "
                "Please install it with `pip install cassio`."
            )
        self.session_id = session_id
        self.ttl_seconds = ttl_seconds

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve all session messages from DB"""
        # The latest are returned, in chronological order
        message_blobs = [
            row["body_blob"].decode("utf-8")  # Decode bytes to string
            for row in self.table.get_partition(
                partition_id=self.session_id,
            )
        ][::-1]
        try:
            items = [json.loads(message_blob) for message_blob in message_blobs]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Write a message to the table"""
        this_row_id = uuid.uuid1()
        self.table.put(
            partition_id=self.session_id,
            row_id=this_row_id,
            body_blob=json.dumps(message_to_dict(message)).encode("utf-8"),  # Encode string to bytes
            ttl_seconds=self.ttl_seconds,
        )

    def clear(self, session_id: str) -> None:
        """Clear session memory from DB"""
        self.table.delete_partition(session_id)
