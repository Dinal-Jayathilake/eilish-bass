from collections.abc import Iterator
from typing import Any

import boto3
from pydantic import BaseModel, Field, PrivateAttr

from app.models.bedrock_chat import (
    ConversationParams,
    Message,
    MessageRole,
    SupportedModel,
)


class BedrockChatClient(BaseModel):
    """High-level abstraction for AWS Bedrock Runtime converse API."""

    region_name: str = Field(default="ap-south-1", description="AWS region")
    profile_name: str | None = Field(None, description="AWS profile name")
    conversation_history: list[Message] = Field(default_factory=list, description="Conversation history")

    _client: Any = PrivateAttr()

    class Config:
        """Configuration class for Pydantic model settings."""

        arbitrary_types_allowed = True

    def model_post_init(self, _context: object) -> None:
        """Initialize the Bedrock client after model initialization."""
        session = boto3.Session(profile_name=self.profile_name)
        self._client = session.client("bedrock-runtime", region_name=self.region_name)

    def create_message(self, role: MessageRole, text: str | None = None) -> Message:
        """Create a new message."""
        message = Message(role=role)
        if text:
            message.add_text(text)
        return message

    def add_message(self, message: Message) -> "BedrockChatClient":
        """Add a message to conversation history."""
        self.conversation_history.append(message)
        return self

    def add_user_message(self, text: str) -> "BedrockChatClient":
        """Add a user message with text."""
        message = Message(role=MessageRole.USER).add_text(text)
        self.conversation_history.append(message)
        return self

    def add_assistant_message(self, text: str) -> "BedrockChatClient":
        """Add an assistant message with text."""
        message = Message(role=MessageRole.ASSISTANT).add_text(text)
        self.conversation_history.append(message)
        return self

    def clear_history(self) -> "BedrockChatClient":
        """Clear conversation history."""
        self.conversation_history.clear()
        return self

    @staticmethod
    def get_supported_models() -> list[SupportedModel]:
        """Get list of supported model identifiers."""
        return list(SupportedModel)

    def _build_request(self, params: ConversationParams) -> dict[str, Any]:
        """Build the request dictionary for Bedrock API calls."""
        messages = params.messages if params.messages is not None else self.conversation_history

        request = {
            "modelId": params.model_id,
            "messages": [msg.to_dict() for msg in messages],
        }

        if params.system:
            request["system"] = params.system

        if params.inference_config:
            request["inferenceConfig"] = params.inference_config.to_dict()

        if params.tool_config:
            request["toolConfig"] = {"tools": [tool.to_dict() for tool in params.tool_config]}

        if params.additional_model_request_fields:
            request["additionalModelRequestFields"] = params.additional_model_request_fields

        if params.additional_model_response_field_paths:
            request["additionalModelResponseFieldPaths"] = params.additional_model_response_field_paths

        return request

    def _process_response_message(self, response: dict[str, Any], messages: list[Message]) -> None:
        """Process and add assistant response to conversation history."""
        if messages is not self.conversation_history or "output" not in response:
            return

        output = response["output"]
        if "message" not in output:
            return

        msg_content = output["message"]["content"]
        assistant_msg = Message(role=MessageRole.ASSISTANT)

        for content in msg_content:
            if "text" in content:
                assistant_msg.add_text(content["text"])
            elif "toolUse" in content:
                tool_use = content["toolUse"]
                assistant_msg.add_tool_use(
                    tool_use_id=tool_use["toolUseId"],
                    name=tool_use["name"],
                    input_data=tool_use["input"],
                )

        self.conversation_history.append(assistant_msg)

    def _process_streaming_response(
        self, messages: list[Message], assistant_message_content: list[dict[str, Any]]
    ) -> None:
        """Process streaming response and add to conversation history."""
        if messages is not self.conversation_history or not assistant_message_content:
            return

        assistant_msg = Message(role=MessageRole.ASSISTANT)
        full_text = "".join([content.get("text", "") for content in assistant_message_content])
        if full_text:
            assistant_msg.add_text(full_text)
            self.conversation_history.append(assistant_msg)

    def converse(self, params: ConversationParams) -> dict[str, Any]:
        """Send a conversation request to Bedrock.

        Args:
            params: Conversation parameters including model_id, messages, etc.

        Returns:
            The response from Bedrock
        """
        request = self._build_request(params)
        response = self._client.converse(**request)

        used_messages = params.messages if params.messages is not None else self.conversation_history
        self._process_response_message(response, used_messages)
        return response

    def converse_stream(self, params: ConversationParams) -> Iterator[dict[str, Any]]:
        """Send a streaming conversation request to Bedrock.

        Args:
            params: Conversation parameters including model_id, messages, etc.

        Yields:
            Streaming response chunks from Bedrock
        """
        request = self._build_request(params)
        response = self._client.converse_stream(**request)

        used_messages = params.messages if params.messages is not None else self.conversation_history
        assistant_message_content = []

        for chunk in response["stream"]:
            yield chunk

            if used_messages is self.conversation_history and "contentBlockDelta" in chunk:
                delta = chunk["contentBlockDelta"]
                if "text" in delta:
                    assistant_message_content.append({"text": delta["text"]})

        self._process_streaming_response(used_messages, assistant_message_content)
