from langchain_core.chat_history import ChatMessageHistory

class FileChatMessageHistory(ChatMessageHistory):
    """A chat message history that is saved to and loaded from a file."""

    @classmethod
    def load(cls, file_path: str):
        """Load the chat history from a file."""
        # Implement the loading logic here.
        pass

    def save(self, file_path: str):
        """Save the chat history to a file."""
        # Implement the saving logic here.
        pass


__all__ = [
    "FileChatMessageHistory",
]

