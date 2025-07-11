import os
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

from app.services.logger import logger

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(os.environ["WEBSOCKET_CONNECTIONS_TABLE"])


def lambda_handler(event: dict[str, Any], _context: dict[str, Any]) -> dict[str, Any]:
    """Handle WebSocket connection and disconnection events.

    Processes incoming WebSocket connection ($connect) and disconnection
    ($disconnect) events, managing connection state in DynamoDB.

    Args:
        event: The Lambda event containing WebSocket request context.
        _context: The Lambda context object (unused).

    Returns:
        A response with statusCode indicating success or failure.
    """
    route_key = event["requestContext"]["routeKey"]
    connection_id = event["requestContext"]["connectionId"]

    try:
        if route_key == "$connect":
            return handle_connect(connection_id, event)
        if route_key == "$disconnect":
            return handle_disconnect(connection_id)
        logger.error("Unknown route: %s", route_key)

    except (KeyError, ClientError):
        logger.exception("Error in connection handler")
        return {"statusCode": 500}
    else:
        return {"statusCode": 400}


def handle_connect(connection_id: str, event: dict[str, Any]) -> dict[str, Any]:
    """Store connection information when a client connects.

    Extracts connection metadata from the event and stores it in DynamoDB
    with a 24-hour TTL for automatic cleanup.

    Args:
        connection_id: The unique WebSocket connection identifier.
        event: The connection event containing metadata.

    Returns:
        A response indicating whether the connection was stored successfully.
    """
    try:
        request_context = event["requestContext"]
        current_time = int(time.time())

        table.put_item(
            Item={
                "connectionId": connection_id,
                "timestamp": current_time,
                "connectedAt": request_context.get("connectedAt"),
                "sourceIp": request_context.get("identity", {}).get("sourceIp"),
                "userAgent": request_context.get("identity", {}).get("userAgent"),
                "ttl": current_time + 86400,  # 24 hours TTL
            }
        )

        logger.info("Connection stored: %s", connection_id)

    except (KeyError, ClientError):
        logger.exception("Error storing connection %s", connection_id)
        return {"statusCode": 500}
    else:
        return {"statusCode": 200}


def handle_disconnect(connection_id: str) -> dict[str, Any]:
    """Remove connection information when a client disconnects.

    Cleans up the connection record from DynamoDB when a WebSocket
    connection is terminated.

    Args:
        connection_id: The unique WebSocket connection identifier to remove.

    Returns:
        A response indicating whether the connection was removed successfully.
    """
    try:
        table.delete_item(Key={"connectionId": connection_id})

        logger.info("Connection removed: %s", connection_id)

    except (KeyError, ClientError):
        logger.exception("Error removing connection %s", connection_id)
        return {"statusCode": 500}
    else:
        return {"statusCode": 200}
