import os
import time
import uuid
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
            return handle_connect(connection_id)
        if route_key == "$disconnect":
            return handle_disconnect(connection_id)
        logger.error("Unknown route: %s", route_key)

    except (KeyError, ClientError):
        logger.exception("Error in connection handler")
        return {"statusCode": 500}
    else:
        return {"statusCode": 400}


def handle_connect(connection_id: str) -> dict[str, Any]:
    """Store new connection with a unique session ID.

    Generates a new UUID for the session and stores the mapping between
    the connectionId and sessionId in DynamoDB.

    Args:
        connection_id: The unique WebSocket connection identifier.

    Returns:
        A response indicating whether the connection was stored successfully.
    """
    session_id = str(uuid.uuid4())
    current_time = int(time.time())

    try:
        table.put_item(
            Item={
                "connectionId": connection_id,
                "sessionId": session_id,
                "createdAt": current_time,
                "ttl": current_time + 86400,  # 24 hours TTL
            }
        )
        logger.info("Connection stored: %s with sessionId: %s", connection_id, session_id)

    except ClientError:
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

    except ClientError:
        logger.exception("Error removing connection %s", connection_id)
        return {"statusCode": 500}
    else:
        return {"statusCode": 200}
