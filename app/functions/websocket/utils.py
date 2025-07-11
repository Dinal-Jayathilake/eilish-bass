import json
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError

from app.services.logger import logger


def send_message(connection_id: str, message: dict[str, Any], api_endpoint: str) -> bool:
    """Send a message to a specific WebSocket connection.

    Handles sending data and cleans up stale connections from DynamoDB if they
    are no longer active.

    Args:
        connection_id: The target WebSocket connection identifier.
        message: The message data to send.
        api_endpoint: The API Gateway endpoint URL for posting messages.

    Returns:
        True if the message was sent successfully, False otherwise.
    """
    try:
        https_api_endpoint = f"https://{api_endpoint}"
        client = boto3.client("apigatewaymanagementapi", endpoint_url=https_api_endpoint)
        client.post_to_connection(ConnectionId=connection_id, Data=json.dumps(message))

    except ClientError as e:
        if e.response["Error"]["Code"] == "GoneException":
            logger.warning("Stale connection %s, removing from table.", connection_id)
            _cleanup_connection(connection_id)
        else:
            logger.exception("Failed to send message to %s", connection_id)
        return False
    else:
        return True


def _cleanup_connection(connection_id: str) -> None:
    """Remove a single connection record from DynamoDB.

    Args:
        connection_id: The connection ID to remove.
    """
    try:
        dynamodb = boto3.resource("dynamodb")
        table = dynamodb.Table(os.environ["WEBSOCKET_CONNECTIONS_TABLE"])
        table.delete_item(Key={"connectionId": connection_id})
        logger.info("Cleaned up stale connection: %s", connection_id)

    except ClientError:
        logger.exception("Error cleaning up connection %s", connection_id)
