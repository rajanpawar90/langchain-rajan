"""Playwright browser toolkit for Langchain Community."""

from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit


class PlayWrightBrowserToolkitWrapper:
    """Wrapper class for PlayWrightBrowserToolkit."""

    def __init__(self):
        """Initialize the wrapper instance."""
        self.toolkit = PlayWrightBrowserToolkit()

    def __call__(self, *args, **kwargs):
        """Delegate method call to the PlayWrightBrowserToolkit instance."""
        return self.toolkit(*args, **kwargs)


__all__ = ["PlayWrightBrowserToolkitWrapper"]
