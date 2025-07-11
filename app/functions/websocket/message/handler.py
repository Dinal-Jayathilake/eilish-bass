import json
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError

from app.services.logger import logger

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(os.environ["WEBSOCKET_CONNECTIONS_TABLE"])


def lambda_handler(event: dict[str, Any], _context: dict[str, Any]) -> dict[str, Any]:
    """Handle WebSocket messages and route them appropriately.

    Processes incoming WebSocket messages, parses the action, and routes
    to the appropriate handler function for message processing.

    Args:
        event: The Lambda event containing WebSocket message data.
        _context: The Lambda context object (unused).

    Returns:
        A response indicating the result of message processing.
    """
    route_key = event["requestContext"]["routeKey"]
    connection_id = event["requestContext"]["connectionId"]

    try:
        body = json.loads(event.get("body", "{}"))
        action = body.get("action", route_key)

        if action == "sendMessage":
            return handle_send_message(event, body)
        if action == "broadcast":
            return handle_broadcast(event, body)
        if route_key == "$default":
            return handle_default_route(event)
        return send_to_connection(connection_id, {"error": f"Unknown action: {action}"}, event)

    except json.JSONDecodeError:
        logger.error("Invalid JSON in message body")
        return {"statusCode": 400}
    except (KeyError, ClientError):
        logger.exception("Error in message handler")
        return {"statusCode": 500}


def handle_send_message(event: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """Send a message to a specific connection or broadcast to all.

    Processes sendMessage actions by either sending to a target connection
    or broadcasting to all connected clients if no target is specified.

    Args:
        event: The WebSocket event containing sender information.
        body: The message body containing message content and optional target.

    Returns:
        A response indicating whether the message was sent successfully.
    """
    connection_id = event["requestContext"]["connectionId"]
    message = body.get("message", "")
    target_connection = body.get("targetConnection")

    if target_connection:
        return send_to_connection(
            target_connection, {"type": "message", "from": connection_id, "message": message}, event
        )
    return broadcast_message(
        {"type": "message", "from": connection_id, "message": message}, event, exclude_connection=connection_id
    )


def handle_broadcast(event: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """Broadcast a message to all connected clients.

    Processes broadcast actions by sending the message to all active
    WebSocket connections.

    Args:
        event: The WebSocket event containing sender information.
        body: The message body containing broadcast content.

    Returns:
        A response indicating whether the broadcast was successful.
    """
    message = body.get("message", "")
    connection_id = event["requestContext"]["connectionId"]

    return broadcast_message({"type": "broadcast", "from": connection_id, "message": message}, event)


def handle_default_route(event: dict[str, Any]) -> dict[str, Any]:
    """Handle default route by sending connection info back to client.

    Processes default route messages by retrieving connection information
    from DynamoDB and sending it back to the requesting client.

    Args:
        event: The WebSocket event containing connection information.

    Returns:
        A response containing the connection information or error status.
    """
    connection_id = event["requestContext"]["connectionId"]

    try:
        response = table.get_item(Key={"connectionId": connection_id})
        connection_info = response.get("Item", {})

        return send_to_connection(
            connection_id, {"type": "info", "connectionId": connection_id, "info": connection_info}, event
        )

    except ClientError:
        logger.exception("Error getting connection info")
        return {"statusCode": 500}


def get_api_gateway_client(event: dict[str, Any]) -> boto3.client:
    """Create API Gateway Management API client with proper endpoint.

    Constructs the correct endpoint URL from the event context and returns
    a configured API Gateway Management client for sending messages.

    Args:
        event: The WebSocket event containing domain and stage information.

    Returns:
        A configured API Gateway Management API client.
    """
    endpoint_url = f"https://{event['requestContext']['domainName']}/{event['requestContext']['stage']}"
    return boto3.client("apigatewaymanagementapi", endpoint_url=endpoint_url)


def send_to_connection(connection_id: str, data: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """Send data to a specific WebSocket connection.

    Attempts to send JSON data to a WebSocket connection using the API Gateway
    Management API. Handles stale connections by removing them from DynamoDB.

    Args:
        connection_id: The target connection identifier.
        data: The data to send to the connection.
        event: The WebSocket event for creating the API client.

    Returns:
        A response indicating success, failure, or stale connection cleanup.
    """
    try:
        apigw_client = get_api_gateway_client(event)

        apigw_client.post_to_connection(ConnectionId=connection_id, Data=json.dumps(data))

        logger.info("Message sent to connection: %s", connection_id)

    except ClientError as e:
        error_code = e.response["Error"]["Code"]

        if error_code == "GoneException":
            logger.info("Stale connection found, removing: %s", connection_id)
            try:
                table.delete_item(Key={"connectionId": connection_id})
            except ClientError:
                logger.exception("Error cleaning up stale connection")
            return {"statusCode": 410}
        logger.exception("Error sending message to %s", connection_id)
        return {"statusCode": 500}
    else:
        return {"statusCode": 200}


def broadcast_message(
    data: dict[str, Any], event: dict[str, Any], exclude_connection: str | None = None
) -> dict[str, Any]:
    """Broadcast a message to all active connections.

    Retrieves all active connections from DynamoDB and sends the message
    to each one. Automatically cleans up stale connections that fail.

    Args:
        data: The message data to broadcast.
        event: The WebSocket event for creating the API client.
        exclude_connection: Optional connection ID to exclude from broadcast.

    Returns:
        A response indicating the broadcast results with success/failure counts.
    """
    try:
        connections = get_all_connections()
        apigw_client = get_api_gateway_client(event)

        successful_sends = 0
        failed_sends = 0
        stale_connections = []

        for connection in connections:
            connection_id = connection["connectionId"]

            if exclude_connection and connection_id == exclude_connection:
                continue

            try:
                apigw_client.post_to_connection(ConnectionId=connection_id, Data=json.dumps(data))
                successful_sends += 1

            except ClientError as e:
                if e.response["Error"]["Code"] == "GoneException":
                    # Mark stale connection for cleanup
                    stale_connections.append(connection_id)
                    failed_sends += 1
                else:
                    logger.exception("Error sending to %s", connection_id)
                    failed_sends += 1

        for stale_connection in stale_connections:
            try:
                table.delete_item(Key={"connectionId": stale_connection})
                logger.info("Cleaned up stale connection: %s", stale_connection)
            except ClientError:
                logger.exception("Error cleaning up stale connection %s", stale_connection)

        logger.info("Broadcast complete: %d successful, %d failed", successful_sends, failed_sends)

    except ClientError:
        logger.exception("Error in broadcast")
        return {"statusCode": 500}
    else:
        return {"statusCode": 200}


def get_all_connections() -> list[dict[str, Any]]:
    """Retrieve all active connections from DynamoDB.

    Scans the connections table to get all currently active WebSocket
    connections for broadcasting purposes.

    Returns:
        A list of connection items from DynamoDB, or empty list on error.
    """
    try:
        response = table.scan()
        return response.get("Items", [])
    except ClientError:
        logger.exception("Error scanning connections table")
        return []
