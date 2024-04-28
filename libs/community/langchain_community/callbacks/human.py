from typing import Callable, Dict, Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler


def default_approve(input_: str) -> bool:
    """Ask the user for approval of the given input.

    Args:
        input_ (str): The input to approve.

    Returns:
        bool: True if the input is approved, False otherwise.
    """
    msg = (
        "Do you approve of the following input? "
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no."
    )
    msg += "\n\n" + input_ + "\n"
    resp = input(msg)
    return resp.lower() in ("yes", "y")


async def async_default_approve(input_: str) -> bool:
    """Asynchronously ask the user for approval of the given input.

    Args:
        input_ (str): The input to approve.

    Returns:
        bool: True if the input is approved, False otherwise.
    """
    msg = (
        "Do you approve of the following input? "
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no."
    )
    msg += "\n\n" + input_ + "\n"
    resp = input(msg)
    return resp.lower() in ("yes", "y")


def default_true(_: Dict[str, Any]) -> bool:
    """A function that always returns True.

    Args:
        obj (Any): The object to ignore.

    Returns:
        bool: Always True.
    """
    return True


class HumanRejectedException(Exception):
    """Exception to raise when a person manually review and rejects a value."""


class HumanApprovalCallbackHandler(BaseCallbackHandler):
    """Callback for manually validating values.

    This class provides a way to manually validate values by asking the user for approval.
    If the user rejects the value, a HumanRejectedException is raised.

    Args:
        approve (Callable[[Any], bool]): A function that takes an input and returns a boolean indicating approval.
        should_check (Callable[[Dict[str, Any]], bool]): A function that takes a serialized object and returns a boolean indicating whether to check for approval.
        raise_error (bool): Whether to raise a HumanRejectedException if the user rejects the value. Defaults to True.
    """

    raise_error: bool = True

    def __init__(
        self,
        approve: Callable[[str], bool] = default_approve,
        should_check: Callable[[Dict[str, Any]], bool] = default_true,
    ):
        self._approve = approve
        self._should_check = should_check

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if self._should_check(serialized) and not self._approve(input_str):
            if self.raise_error:
                raise HumanRejectedException(
                    f"Inputs {input_str} to tool {serialized} were rejected."
                )
            else:
                return False


class AsyncHumanApprovalCallbackHandler(AsyncCallbackHandler):
    """Asynchronous callback for manually validating values.

    This class provides an asynchronous way to manually validate values by asking the user for approval.
    If the user rejects the value, a HumanRejectedException is raised.

    Args:
        approve (Callable[[Any], Awaitable[bool]]): An asynchronous function that takes an input and returns a boolean indicating approval.
        should_check (Callable[[Dict[str, Any]], bool]): A function that takes a serialized object and returns a boolean indicating whether to check for approval.
        raise_error (bool): Whether to raise a HumanRejectedException if the user rejects the value. Defaults to True.
    """

    raise_error: bool = True

    def __init__(
        self,
        approve: Callable[[str], Awaitable[bool]] = async_default_approve,
        should_check: Callable[[Dict[str, Any]], bool] = default_true,
    ):
        self._approve = approve
        self._should_check = should_check

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if self._should_check(serialized) and not await self._approve(input_str):
            if self.raise_error:
                raise HumanRejectedException(
                    f"Inputs {input_str} to tool {serialized} were rejected."
                )
            else:
                return False
