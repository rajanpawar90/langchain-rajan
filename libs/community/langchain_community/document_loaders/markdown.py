import importlib.metadata
from typing import List

from langchain_community.document_loaders.unstructured import UnstructuredFileLoader


class UnstructuredMarkdownLoader(UnstructuredFileLoader):
    """Load `Markdown` files using `Unstructured`.

    You can run the loader in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredMarkdownLoader

    loader = UnstructuredMarkdownLoader(
        "example.md", mode="elements", strategy="fast",
    )
    docs = loader.load()

    References
    ----------
    https://unstructured-io.github.io/unstructured/core/partition.html#partition-md
    """

    def __init_subclass__(cls, **kwargs):
        if cls is not UnstructuredMarkdownLoader:
            return

        if not hasattr(cls, "_get_elements"):
            raise ValueError(
                "Subclasses of UnstructuredMarkdownLoader must override the `_get_elements` method."
            )

    def _get_elements(self) -> List[str]:
        unstructured_version = tuple(map(int, importlib.metadata.version("unstructured").split(".")))

        if unstructured_version < (0, 4, 16):
            raise ValueError(
                f"You are on unstructured version {importlib.metadata.version('unstructured')}. "
                "Partitioning markdown files is only supported in unstructured>=0.4.16."
            )

        partition_md = importlib.import_module("unstructured.partition.md").partition_md
        return partition_md(filename=self.file_path, **self.unstructured_kwargs)
