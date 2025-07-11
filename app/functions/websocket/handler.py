import json
from typing import Any

from app.services.logger import logger


def lambda_handler(event: dict[str, Any], _context: dict[str, Any]) -> dict[str, Any]:
    """
    Handles default WebSocket messages, echoing the received message directly
    back to the sender using a route response.
    """
    connection_id = event["requestContext"]["connectionId"]
    logger.info("Received message on $default route from %s", connection_id)

    try:
        body = event.get("body", "{}")
    except Exception:
        logger.exception("Error in default message handler for %s", connection_id)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "An internal error occurred."}),
        }
    else:
        return {
            "statusCode": 200,
            "body": body,
        }
