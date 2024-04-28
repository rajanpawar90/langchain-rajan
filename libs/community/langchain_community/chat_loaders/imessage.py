from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Iterator, List, Optional, Union

import pathlib

from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from sqlite3 import Cursor


def raise_if_not_exists(file_path: pathlib.Path, msg: str):
    if not file_path.exists():
        raise FileNotFoundError(msg)


def nanoseconds_from_2001_to_datetime(nanoseconds: int) -> datetime:
    # Convert nanoseconds to seconds (1 second = 1e9 nanoseconds)
    timestamp_in_seconds = nanoseconds / 1e9

    # The reference date is January 1, 2001, in Unix time
    reference_date_seconds = datetime(2001, 1, 1).timestamp()

    # Calculate the actual timestamp by adding the reference date
    actual_timestamp = reference_date_seconds + timestamp_in_seconds

    # Convert to a datetime object
    return datetime.fromtimestamp(actual_timestamp)


class IMessageChatLoader(BaseChatLoader):
    """Load chat sessions from the `iMessage` chat.db SQLite file."""

    def __init__(self, path: Optional[Union[str, pathlib.Path]] = None):
        """
        Initialize the IMessageChatLoader.

        Args:
            path (str or Path, optional): Path to the chat.db SQLite file.
                Defaults to None, in which case the default path
                ~/Library/Messages/chat.db will be used.
        """
        if path is None:
            path = pathlib.Path.home() / "Library" / "Messages" / "chat.db"
        self.db_path = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
        self.raise_if_not_exists(self.db_path, f"File {self.db_path} not found")

        try:
            import sqlite3  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The sqlite3 module is required to load iMessage chats.\n"
                "Please install it with `pip install pysqlite3`"
            ) from e

    def _parse_attributedBody(self, attributedBody: bytes) -> str:
        # ... (same as original)

    def _get_session_query(self, use_chat_handle_table: bool) -> str:
        # ... (same as original)

    def _load_single_chat_session(
        self, cursor: Cursor, use_chat_handle_table: bool, chat_id: int
    ) -> ChatSession:
        # ... (same as original)

    def lazy_load(self) -> Iterator[ChatSession]:
        """
        Lazy load the chat sessions from the iMessage chat.db
        and yield them in the required format.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # See if chat_handle_join table exists:
            query = """SELECT name FROM sqlite_master
                       WHERE type='table' AND name='chat_handle_join';"""

            cursor.execute(query)
            is_chat_handle_join_exists = bool(cursor.fetchone())

            # Fetch the list of chat IDs sorted by time (most recent first)
            query = """SELECT chat_id
            FROM message
            JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
            GROUP BY chat_id
            ORDER BY MAX(date) DESC;"""
            cursor.execute(query)
            chat_ids = [row[0] for row in cursor.fetchall()]

            for chat_id in chat_ids:
                yield self._load_single_chat_session(
                    cursor, is_chat_handle_join_exists, chat_id
                )

    def load_chat_sessions(self) -> List[ChatSession]:
        """Load all chat sessions at once."""
        chat_sessions = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # See if chat_handle_join table exists:
            query = """SELECT name FROM sqlite_master
                       WHERE type='table' AND name='chat_handle_join';"""

            cursor.execute(query)
            is_chat_handle_join_exists = bool(cursor.fetchone())

            # Fetch the list of chat IDs sorted by time (most recent first)
            query = """SELECT chat_id
            FROM message
            JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
            GROUP BY chat_id
            ORDER BY MAX(date) DESC;"""
            cursor.execute(query)
            chat_ids = [row[0] for row in cursor.fetchall()]

            for chat_id in chat_ids:
                chat_sessions.append(
                    self._load_single_chat_session(
                        cursor, is_chat_handle_join_exists, chat_id
                    )
                )
        return chat_sessions

    def __repr__(self):
        return f"<IMessageChatLoader db_path={self.db_path}>"
