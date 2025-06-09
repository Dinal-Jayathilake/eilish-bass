import logging
import sys

import structlog

logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)


def format_event_message(_logger: object, _method_name: object, event_dict: dict) -> dict:
    """Moves the 'event' field to 'message' field in the event dictionary.

    Args:
        _logger: Unused logger parameter.
        _method_name: Unused method name parameter.
        event_dict: Dictionary containing event data.

    Returns:
        Modified event dictionary with 'event' field moved to 'message'.
    """
    event = event_dict.pop("event", "")
    event_dict["message"] = event
    return event_dict


structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        format_event_message,
        structlog.processors.JSONRenderer(indent=2),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
