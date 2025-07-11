import json
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError

from app.services.logger import logger


def send_message_to_connection(connection_id: str, message: dict[str, Any], api_endpoint: str) -> bool:
    """Utility function to send a message to a specific connection.

    Creates an API Gateway Management client and attempts to send a message
    to the specified WebSocket connection.

    Args:
        connection_id: The target WebSocket connection identifier.
        message: The message data to send.
        api_endpoint: The API Gateway endpoint URL.

    Returns:
        True if the message was sent successfully, False otherwise.
    """
    try:
        client = boto3.client("apigatewaymanagementapi", endpoint_url=api_endpoint)

        client.post_to_connection(ConnectionId=connection_id, Data=json.dumps(message))

    except ClientError:
        logger.exception("Failed to send message to %s", connection_id)
        return False
    else:
        return True


def get_active_connections() -> list[str]:
    """Get list of active connection IDs.

    Scans the WebSocket connections table to retrieve all currently
    active connection identifiers.

    Returns:
        A list of active connection IDs, or empty list on error.
    """
    try:
        dynamodb = boto3.resource("dynamodb")
        table = dynamodb.Table(os.environ["WEBSOCKET_CONNECTIONS_TABLE"])

        response = table.scan(ProjectionExpression="connectionId")

        return [item["connectionId"] for item in response.get("Items", [])]

    except ClientError:
        logger.exception("Error getting active connections")
        return []


def cleanup_stale_connections(stale_connection_ids: list[str]) -> None:
    """Remove stale connections from DynamoDB.

    Batch deletes multiple connection records from the DynamoDB table
    to clean up connections that are no longer active.

    Args:
        stale_connection_ids: List of connection IDs to remove.
    """
    if not stale_connection_ids:
        return

    try:
        dynamodb = boto3.resource("dynamodb")
        table = dynamodb.Table(os.environ["WEBSOCKET_CONNECTIONS_TABLE"])

        with table.batch_writer() as batch:
            for connection_id in stale_connection_ids:
                batch.delete_item(Key={"connectionId": connection_id})

        logger.info("Cleaned up %d stale connections", len(stale_connection_ids))

    except ClientError:
        logger.exception("Error cleaning up stale connections")
