import oracledb
from typing import Any, Dict, List, Union

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

class OracleAutonomousDatabaseLoader(BaseLoader):
    """
    Load from oracle adb

    Autonomous Database connection can be made by either connection_string
    or tns name. wallet_location and wallet_password are required
    for TLS connection.
    Each document will represent one row of the query result.
    Columns are written into the `page_content` and 'metadata' in
    constructor is written into 'metadata' of document,
    by default, the 'metadata' is None.
    """

    def __init__(
        self,
        query: str,
        user: str,
        password: str,
        schema: Union[str, None] = None,
        tns_name: Union[str, None] = None,
        config_dir: Union[str, None] = None,
        wallet_location: Union[str, None] = None,
        wallet_password: Union[str, None] = None,
        connection_string: Union[str, None] = None,
        metadata: Union[List[str], None] = None,
    ):
        """
        init method
        :param query: sql query to execute
        :param user: username
        :param password: user password
        :param schema: schema to run in database
        :param tns_name: tns name in tnsname.ora
        :param config_dir: directory of config files(tnsname.ora, wallet)
        :param wallet_location: location of wallet
        :param wallet_password: password of wallet
        :param connection_string: connection string to connect to adb instance
        :param metadata: metadata used in document
        """
        self._validate_arguments(query, user, password)

        self.query = query
        self.user = user
        self.password = password
        self.schema = schema
        self.tns_name = tns_name
        self.config_dir = config_dir
        self.wallet_location = wallet_location
        self.wallet_password = wallet_password
        self.connection_string = connection_string
        self.metadata = metadata

        self.dsn: Union[str, None]
        self._set_dsn()

    def _validate_arguments(self, query: str, user: str, password: str) -> None:
        if not query or not user or not password:
            raise ValueError("query, user, and password cannot be None or empty")

    def _set_dsn(self) -> None:
        if self.connection_string:
            self.dsn = self.connection_string
        elif self.tns_name:
            self.dsn = self.tns_name
        else:
            self.dsn = None

    def _get_connection_details(self) -> dict[str, str]:
        connect_param = {
            "user": self.user,
            "password": self.password,
            "dsn": self.dsn,
        }

        if self.dsn == self.tns_name:
            connect_param["config_dir"] = self.config_dir

        if self.wallet_location and self.wallet_password:
            connect_param["wallet_location"] = self.wallet_location
            connect_param["wallet_password"] = self.wallet_password

        return connect_param

    def _run_query(self) -> List[Dict[str, Any]]:
        try:
            connection_details = self._get_connection_details()
            connection = oracledb.connect(**connection_details)
            cursor = connection.cursor()

            if self.schema:
                cursor.execute(f"alter session set current_schema={self.schema}")
            cursor.execute(self.query)

            columns = [col[0] for col in cursor.description]
            data = cursor.fetchall()
            data = [dict(zip(columns, row)) for row in data]

        except oracledb.DatabaseError as e:
            import logging

            logging.error(f"Got error while connecting: {str(e)}")
            data = []
        finally:
            cursor.close()
            connection.close()

        return data

    def load(self) -> List[Document]:
        data = self._run_query()
        documents = []
        metadata_columns = self.metadata if self.metadata else []

        for row in data:
            metadata = {
                key: value for key, value in row.items() if key in metadata_columns
            }
            doc = Document(page_content=str(row), metadata=metadata)
            documents.append(doc)

        return documents

    def __repr__(self) -> str:
        return (
            f"OracleAutonomousDatabaseLoader("
            f"query={self.query!r}, "
            f"user={self.user!r}, "
            f"password={self.password!r}, "
            f"schema={self.schema!r}, "
            f"tns_name={self.tns_name!r}, "
            f"config_dir={self.config_dir!r}, "
            f"wallet_location={self.wallet_location!r}, "
            f"wallet_password={self.wallet_password!r}, "
            f"connection_string={self.connection_string!r}, "
            f"metadata={self.metadata!r})"
        )
