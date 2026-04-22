"""
Neatlogs - LLM Call Tracking Library
==========================================

A comprehensive LLM tracking system.
Automatically captures and logs all LLM API calls with detailed metrics.
"""

from .core import get_tracker, LLMTracker
from .instrumentation.manager import setup_import_monitor
import logging
import atexit
import threading
from typing import List, Optional, Dict

__version__ = "1.1.8"
__all__ = ['init', 'get_tracker', 'add_tags', 'get_session_stats', 'get_langchain_callback_handler']

# --- Global Tracker Instance and Initialization ---

_global_tracker: Optional[LLMTracker] = None
_init_lock = threading.Lock()


def init(
    api_key: str,
    tags: Optional[List[str]] = None,
    debug: bool = False
):
    """
    Initialize the Neatlogs tracking system.


    Args:
        api_key (str): API key for the session. Will be persisted and logged.
        tags (List[str], optional): List of tags to associate with the tracking session.
        debug (bool): Enable debug logging. Defaults to False.

    Returns:
        LLMTracker: The initialized tracker instance.

    Example:
        >>> import neatlogs
        >>> tracker = neatlogs.init(
        ...     api_key="your_api_key",
        ...     tags=["tag1", "tag2"]
        ... )
        >>> # Now all calls are automatically tracked!
    """

    session_id = None
    agent_id = None
    thread_id = None

    global _global_tracker

    if debug:
        logging.basicConfig(level=logging.DEBUG)

    with _init_lock:
        if _global_tracker is None:
            _global_tracker = LLMTracker(
                api_key=api_key,
                session_id=session_id,
                agent_id=agent_id,
                thread_id=thread_id,
                tags=tags,
            )
            from .instrumentation import manager
            manager.instrument_all(_global_tracker)

            # Log initialization info
            logging.info("🚀 Neatlogs Tracker initialized successfully!")
            logging.info(f"   📊 Session: {_global_tracker.session_id}")
            logging.info(f"   🤖 Agent: {_global_tracker.agent_id}")
            logging.info(f"   🧵 Thread: {_global_tracker.thread_id}")
            if tags:
                logging.info(f"   🏷️  Tags: {tags}")

    return _global_tracker


def get_langchain_callback_handler(api_key: Optional[str] = None, tags: Optional[List[str]] = None):
    """
    Get the LangChain callback handler for Neatlogs tracking.


    This function lazily imports the callback handler to avoid triggering
    framework detection when it's not needed.

    Args:
        api_key (str, optional): API key for the tracker.
        tags (List[str], optional): Tags to associate with the tracking session.

    Returns:
        NeatlogsLangchainCallbackHandler: The callback handler instance.
    """
    from .integration.callbacks.langchain.callback import NeatlogsLangchainCallbackHandler
    return NeatlogsLangchainCallbackHandler(api_key=api_key, tags=tags)


def add_tags(tags: List[str]):
    """
    Add tags to the current Neatlogs tracker.


    Args:
        tags (list): List of tags to add

    Example:
        >>> neatlogs.add_tags(["production", "customer-support", "v2.1"])
    """
    tracker = get_tracker()
    if not tracker:
        raise RuntimeError(
            "Tracker not initialized. Call neatlogs.init() first.")

    tracker.add_tags(tags)


def get_session_stats() -> Dict:
    """
    Get statistics for all LLM calls in the current session.


    Returns statistics including total calls, tokens, cost, and breakdowns
    by provider and model.

    Returns:
        dict: Session statistics with keys:
            - total_calls: Number of LLM calls
            - total_tokens_input: Total input tokens
            - total_tokens_output: Total output tokens
            - total_tokens: Combined total tokens
            - total_cost: Total cost in USD
            - average_response_time: Average response time in seconds
            - provider_breakdown: Stats per provider
            - model_breakdown: Stats per model
            - status_breakdown: Count by status (SUCCESS/FAILURE)

    Example:
        >>> import neatlogs
        >>> neatlogs.init(api_key="key")
        >>> # ... make some LLM calls ...
        >>> stats = neatlogs.get_session_stats()
        >>> print(f"Total calls: {stats['total_calls']}")
        >>> print(f"Total cost: ${stats['total_cost']:.4f}")
    """
    tracker = get_tracker()
    if not tracker:
        return {
            "total_calls": 0,
            "total_tokens_input": 0,
            "total_tokens_output": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
            "provider_breakdown": {},
            "model_breakdown": {},
            "status_breakdown": {},
        }
    return tracker.get_session_stats()


# --- Automatic Instrumentation Setup ---
# This is the core of the "magic". The import hook is set up
# the moment the neatlogs library is imported.
setup_import_monitor()


def _shutdown_neatlogs():
    """Shutdown the Neatlogs tracker and clean up resources on exit."""
    logging.debug("Neatlogs: atexit handler '_shutdown_neatlogs' called.")
    tracker = get_tracker()
    if tracker:
        tracker.shutdown()
    logging.debug("Neatlogs: atexit handler '_shutdown_neatlogs' finished.")


# Ensure that all data is sent and resources are cleaned up on exit.
atexit.register(_shutdown_neatlogs)


# Configure a default handler for the library's logger.
# This prevents "No handler found" warnings if the user of the library
# does not configure logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())
