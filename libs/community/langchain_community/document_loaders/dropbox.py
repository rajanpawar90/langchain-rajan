# Prerequisites:
# 1. Create a Dropbox app.
# 2. Give the app these scope permissions: `files.metadata.read`
#    and `files.content.read`.
# 3. Generate access token: https://www.dropbox.com/developers/apps/create.
# 4. `pip install dropbox` (requires `pip install unstructured[pdf]` for PDF filetype).

import pathlib
from typing import Any, Dict, List, Optional
from functools import lru_cache
from typing import overload
import dataclasses
from dropbox import Dropbox, exceptions
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPDFLoader

@dataclasses.dataclass
class DropboxLoader:
    """Load files from `Dropbox`.

    In addition to common files such as text and PDF files, it also supports
    *Dropbox Paper* files.
    """

    dropbox_access_token: str
    """Dropbox access token."""
    dropbox_folder_path: Optional[str] = None
    """The folder path to load from."""
    dropbox_file_paths: Optional[List[str]] = None
    """The file paths to load from."""
    recursive: bool = False
    """Flag to indicate whether to load files recursively from subfolders."""

    @lru_cache(maxsize=None)
    def _create_dropbox_client(self) -> Dropbox:
        """Create a Dropbox client."""
        try:
            dbx = Dropbox(self.dropbox_access_token)
            dbx.users_get_current_account()
        except exceptions.AuthError as ex:
            raise ValueError(
                "Invalid Dropbox access token. Please verify your token and try again."
            ) from ex
        return dbx

    def _load_file_from_path(self, file_path: pathlib.Path) -> Optional[Document]:
        """Load a file from a Dropbox path."""
        dbx = self._create_dropbox_client()

        try:
            file_metadata = dbx.files_get_metadata(file_path.as_posix())

            if file_metadata.is_downloadable:
                _, response = dbx.files_download(file_path.as_posix())

            # Some types such as Paper, need to be exported.
            elif file_metadata.export_info:
                _, response = dbx.files_export(file_path.as_posix(), "markdown")

        except exceptions.ApiError as ex:
            print(f"Could not load file: {file_path}. Please verify the file path and try again.")
            return None

        try:
            text = response.content.decode("utf-8")
        except UnicodeDecodeError:
            file_extension = file_path.suffix.lower()

            if file_extension == ".pdf":
                print(f"File {file_path} type detected as .pdf")
                temp_pdf = pathlib.Path(tempfile.mktemp()).with_suffix(".pdf")
                temp_pdf.write_bytes(response.content)

                try:
                    loader = UnstructuredPDFLoader(str(temp_pdf))
                    docs = loader.load()
                    if docs:
                        return docs[0]
                except Exception as pdf_ex:
                    print(f"Error while trying to parse PDF {file_path}: {pdf_ex}")
                    return None
            else:
                print(f"File {file_path} could not be decoded as pdf or text. Skipping.")
                return None

        metadata = {
            "source": f"dropbox://{file_path}",
            "title": file_path.name,
        }
        return Document(page_content=text, metadata=metadata)

    @overload
    def load(self) -> List[Document]: ...

    @load.default
    def load(self) -> Optional[Document]:
        if self.dropbox_folder_path is not None:
            folder_path = pathlib.Path(self.dropbox_folder_path)
            return self._load_documents_from_folder(folder_path)
        else:
            return self._load_documents_from_paths()

    def _load_documents_from_folder(self, folder_path: pathlib.Path) -> List[Document]:
        """Load documents from a Dropbox folder."""
        try:
            results = self._create_dropbox_client().files_list_folder(folder_path.as_posix(), recursive=self.recursive)
        except exceptions.ApiError as ex:
            raise ValueError(
                f"Could not list files in the folder: {folder_path}. "
                "Please verify the folder path and try again."
            ) from ex

        files = [entry for entry in results.entries if isinstance(entry, FileMetadata)]
        documents = [
            doc
            for doc in (self._load_file_from_path(file_path) for file_path in files)
            if doc is not None
        ]
        return documents

    def _load_documents_from_paths(self) -> List[Document]:
        """Load documents from a list of Dropbox file paths."""
        if not self.dropbox_file_paths:
            raise ValueError("file_paths must be set")

        return [
            doc
            for doc in (
                self._load_file_from_path(pathlib.Path(file_path))
                for file_path in self.dropbox_file_paths
            )
            if doc is not None
        ]
