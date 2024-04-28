"""**Callback handlers** allow listening to events in LangChain.

**Class hierarchy:**

.. code-block::

    BaseCallbackHandler --> <name>CallbackHandler  # Example: AimCallbackHandler
"""
import importlib

_MODULE_LOOKUP = {
    "aim_callback": "langchain_community.callbacks.aim_callback",
    "argilla_callback": "langchain_community.callbacks.argilla_callback",
    "arize_callback": "langchain_community.callbacks.arize_callback",
    "arthur_callback": "langchain_community.callbacks.arthur_callback",
    "clearml_callback": "langchain_community.callbacks.clearml_callback",
    "comet_ml_callback": "langchain_community.callbacks.comet_ml_callback",
    "context_callback": "langchain_community.callbacks.context_callback",
    "fiddler_callback": "langchain_community.callbacks.fiddler_callback",
    "flyte_callback": "langchain_community.callbacks.flyte_callback",
    "human": "langchain_community.callbacks.human",
    "infino_callback": "langchain_community.callbacks.infino_callback",
    "llmonitor_callback": "langchain_community.callbacks.llmonitor_callback",
    "labelstudio_callback": "langchain_community.callbacks.labelstudio_callback",
    "mlflow_callback": "langchain_community.callbacks.mlflow_callback",
    "openai_info": "langchain_community.callbacks.openai_info",
    "promptlayer_callback": "langchain_community.callbacks.promptlayer_callback",
    "sagemaker_callback": "langchain_community.callbacks.sagemaker_callback",
    "streamlit": "langchain_community.callbacks.streamlit",
    "trubrics_callback": "langchain_community.callbacks.trubrics_callback",
    "uptrain_callback": "langchain_community.callbacks.uptrain_callback",
    "wandb_callback": "langchain_community.callbacks.wandb_callback",
    "whylabs_callback": "langchain_community.callbacks.whylabs_callback",
}


def __getattr__(name: str) -> Any:
    """Dynamically import and return callback handlers.

    Args:
        name (str): The name of the callback handler to import.

    Raises:
        AttributeError: If the specified callback handler does not exist.

    Returns:
        Any: The imported callback handler.
    """
    if name in _MODULE_LOOKUP:
        module = importlib.import_module(_MODULE_LOOKUP[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "aim_callback",
    "argilla_callback",
    "arize_callback",
    "arthur_callback",
    "clearml_callback",
    "comet_ml_callback",
    "context_callback",
    "fiddler_callback",
    "flyte_callback",
    "human",
    "infino_callback",
    "llmonitor_callback",
    "labelstudio_callback",
    "mlflow_callback",
    "openai_info",
    "promptlayer_callback",
    "sagemaker_callback",
    "streamlit",
    "trubrics_callback",
    "uptrain_callback",
    "wandb_callback",
    "whylabs_callback",
]
