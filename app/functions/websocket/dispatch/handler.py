import json
from typing import Any

from app.functions.websocket.utils import send_message
from app.services.logger import logger


def lambda_handler(event: dict[str, Any], _context: dict[str, Any]) -> dict[str, Any]:
    """
    Handles a 'dispatch' route, sending a message to a specified
    recipient using the ApiGatewayManagementApi.
    """
    connection_id = event["requestContext"]["connectionId"]
    api_endpoint = f"{event['requestContext']['domainName']}/{event['requestContext']['stage']}"

    try:
        body = json.loads(event.get("body", "{}"))
        recipient_id = body.get("recipientConnectionId")
        message_content = body.get("message", "No message content")

        if not recipient_id:
            logger.error("recipientConnectionId not provided for dispatch by %s", connection_id)
            return {"statusCode": 400}

        logger.info("Dispatching message from %s to %s", connection_id, recipient_id)
        send_message(recipient_id, {"received": message_content}, api_endpoint)

    except Exception:
        logger.exception("Error in dispatch handler for %s", connection_id)
        return {"statusCode": 500}
    else:
        return {"statusCode": 200}
