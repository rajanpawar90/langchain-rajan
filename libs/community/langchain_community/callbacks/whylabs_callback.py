import os
from typing import Any, Dict, Optional

import whylogs as why
from whylogs.api.writer.whylabs import WhyLabsWriter
from whylogs.experimental.core.udf_schema import udf_schema

import langkit
from langkit.callback_handler import get_callback_instance


class WhyLabsCallbackHandler(langkit.BaseCallbackHandler):
    """
    Callback Handler for logging to WhyLabs. This callback handler utilizes
    `langkit` to extract features from the prompts & responses when interacting with
    an LLM. These features can be used to guardrail, evaluate, and observe interactions
    over time to detect issues relating to hallucinations, prompt engineering,
    or output validation. LangKit is an LLM monitoring toolkit developed by WhyLabs.

    Args:
        logger (Optional[whylogs.logger.Logger]): WhyLabs logger instance.
            Optional because a new logger will be created if not provided.
    """

    def __init__(self, logger: Optional[whylogs.logger.Logger] = None):
        super().__init__()
        self.logger = logger or self._create_logger()

    def _create_logger(self) -> whylogs.logger.Logger:
        api_key = os.getenv("WHYLABS_API_KEY")
        org_id = os.getenv("WHYLABS_DEFAULT_ORG_ID")
        dataset_id = os.getenv("WHYLABS_DEFAULT_DATASET_ID")

        if not all([api_key, org_id, dataset_id]):
            raise ValueError(
                "To use the WhyLabsCallbackHandler, you need to set the following environment variables: WHYLABS_API_KEY, WHYLABS_DEFAULT_ORG_ID, WHYLABS_DEFAULT_DATASET_ID"
            )

        whylabs_writer = WhyLabsWriter(api_key=api_key, org_id=org_id, dataset_id=dataset_id)

        logger = why.logger(
            mode="rolling", interval=5, when="M", schema=udf_schema()
        )

        logger.append_writer(writer=whylabs_writer)
        return logger

    def flush(self) -> None:
        """Explicitly write current profile if using a rolling logger."""
        if self.logger and hasattr(self.logger, "_do_rollover"):
            self.logger._do_rollover()
            diagnostic_logger.info("Flushing WhyLabs logger, writing profile...")

    def close(self) -> None:
        """Close any loggers to allow writing out of any profiles before exiting."""
        if self.logger and hasattr(self.logger, "close"):
            self.logger.close()
            diagnostic_logger.info("Closing WhyLabs logger, see you next time!")

    def __enter__(self) -> "WhyLabsCallbackHandler":
        return self

    def __exit__(
        self, exception_type: Any, exception_value: Any, traceback: Any
    ) -> None:
        self.close()

    @classmethod
    def from_params(
        cls,
        *,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        sentiment: bool = False,
        toxicity: bool = False,
        themes: bool = False,
        logger: Optional[whylogs.logger.Logger] = None,
    ) -> "WhyLabsCallbackHandler":
        """Instantiate whylogs Logger from params.

        Args:
            api_key (Optional[str]): WhyLabs API key. Optional because the preferred
                way to specify the API key is with environment variable
                WHYLABS_API_KEY.
            org_id (Optional[str]): WhyLabs organization id to write profiles to.
                If not set must be specified in environment variable
                WHYLABS_DEFAULT_ORG_ID.
            dataset_id (Optional[str]): The model or dataset this callback is gathering
                telemetry for. If not set must be specified in environment variable
                WHYLABS_DEFAULT_DATASET_ID.
            sentiment (bool): If True will initialize a model to perform
                sentiment analysis compound score. Defaults to False and will not gather
                this metric.
            toxicity (bool): If True will initialize a model to score
                toxicity. Defaults to False and will not gather this metric.
            themes (bool): If True will initialize a model to calculate
                distance to configured themes. Defaults to None and will not gather this
                metric.
            logger (Optional[whylogs.logger.Logger]): If specified will bind the
                configured logger as the telemetry gathering agent. Defaults to LangKit
                schema with periodic WhyLabs writer.
        """
        import_langkit(sentiment=sentiment, toxicity=toxicity, themes=themes)

        callback_handler_cls = get_callback_instance(logger=logger, impl=cls)
        diagnostic_logger.info(
            "Started whylogs Logger with WhyLabsWriter and initialized LangKit. 📝"
        )
        return callback_handler_cls
