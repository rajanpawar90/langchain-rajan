import json
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Literal, Optional

import sqlalchemy as sa
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

try:
    from langchain_core.chat_history import BaseChatMessageHistory
except ImportError:
    from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

logger = logging.getLogger(__name__)

Base = declarative_base()


class BaseMessageConverter(ABC):
    """Convert BaseMessage to the SQLAlchemy model."""

    @abstractmethod
    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        """Convert a SQLAlchemy model to a BaseMessage instance."""
        raise NotImplementedError

    @abstractmethod
    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
        """Convert a BaseMessage instance to a SQLAlchemy model."""
        raise NotImplementedError

    @abstractmethod
    def get_sql_model_class(self) -> Any:
        """Get the SQLAlchemy model class."""
        raise NotImplementedError


def create_message_model(table_name: str, DynamicBase: Any) -> Any:
    """
    Create a message model for a given table name.

    Args:
        table_name: The name of the table to use.
        DynamicBase: The base class to use for the model.

    Returns:
        The model class.

    """

    # Model declared inside a function to have a dynamic table name.
    class Message(DynamicBase):  # type: ignore[valid-type, misc]
        __tablename__ = table_name
        id = sa.Column(sa.Integer, primary_key=True)
        session_id = sa.Column(sa.Text)
        message = sa.Column(sa.Text)

    return Message


class DefaultMessageConverter(BaseMessageConverter):
    """The default message converter for SQLChatMessageHistory."""

    def __init__(self, table_name: str):
        self.model_class = create_message_model(table_name, Base)

    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        return messages_from_dict([json.loads(sql_message.message)])[0]

    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
        return self.model_class(
            session_id=session_id, message=json.dumps(message_to_dict(message))
        )

    def get_sql_model_class(self) -> Any:
        return self.model_class


@sa.event.listens_for(sa.engine.Engine, "connect")
def set_sqlite_pragma(conn, rec):
    if isinstance(conn.dialect.name, sa.dialects.sqlite.SQLiteDialect_py):
        conn.exec("PRAGMA foreign_keys=ON;")


class SQLChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in an SQL database."""

    def __init__(
        self,
        session_id: str,
        connection_string: str,
        table_name: str = "message_store",
        session_id_field_name: str = "session_id",
        custom_message_converter: Optional[BaseMessageConverter] = None,
    ):
        self.connection_string = connection_string
        self.engine = sa.create_engine(connection_string, echo=False)
        self.session_id_field_name = session_id_field_name
        self.converter = custom_message_converter or DefaultMessageConverter(table_name)
        self.sql_model_class = self.converter.get_sql_model_class()
        if not hasattr(self.sql_model_class, session_id_field_name):
            raise ValueError("SQL model class must have session_id column")
        self._create_table_if_not_exists()

        self.session_id = session_id
        self.Session = sessionmaker(bind=self.engine)

    def _create_table_if_not_exists(self) -> None:
        try:
            self.sql_model_class.metadata.create_all(self.engine)
        except NoSuchTableError:
            pass

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve all messages from db"""
        with self.get_session() as session:
            result = (
                session.query(self.sql_model_class)
                .where(
                    getattr(self.sql_model_class, self.session_id_field_name)
                    == self.session_id
                )
                .order_by(self.sql_model_class.id.asc())
            )
            messages = []
            for record in result:
                messages.append(self.converter.from_sql_model(record))
            return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in db"""
        with self.get_session() as session:
            session.add(self.converter.to_sql_model(message, self.session_id))
            session.flush()

    def clear(self) -> None:
        """Clear session memory from db"""

        with self.get_session() as session:
            session.query(self.sql_model_class).filter(
                getattr(self.sql_model_class, self.session_id_field_name)
                == self.session_id
            ).delete()
            session.commit()

    def get_session(self, *args, **kwargs) -> sa.orm.Session:
        """Return a new session instance."""
        return self.Session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the session."""
        self.Session.close()
