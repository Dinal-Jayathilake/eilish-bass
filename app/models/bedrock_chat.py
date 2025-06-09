from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class MessageRole(str, Enum):
    """Message roles for conversation."""

    USER = "user"
    ASSISTANT = "assistant"


class SupportedModel(str, Enum):
    """Supported Bedrock model identifiers."""

    CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_3_5_SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_4_SONNET = "anthropic.claude-4-sonnet-20250514-v1:0"


class ContentType(str, Enum):
    """Content types for messages."""

    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"
    TOOL_USE = "toolUse"
    TOOL_RESULT = "toolResult"


class ImageContent(BaseModel):
    """Image content for messages."""

    format: str = Field(..., description="Image format: png, jpeg, gif, webp")
    source: bytes = Field(..., description="Image bytes")

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate image format is one of the allowed formats."""
        allowed_formats = {"png", "jpeg", "gif", "webp"}
        if v.lower() not in allowed_formats:
            error_msg = f"Format must be one of: {allowed_formats}"
            raise ValueError(error_msg)
        return v.lower()

    def to_dict(self) -> dict[str, Any]:
        """Convert to API format."""
        return {
            "image": {
                "format": self.format,
                "source": {"bytes": self.source},
            }
        }


class DocumentContent(BaseModel):
    """Document content for messages."""

    format: str = Field(
        ...,
        description="Document format: pdf, csv, doc, docx, xls, xlsx, html, txt, md",
    )
    name: str = Field(..., description="Document name")
    source: bytes = Field(..., description="Document bytes")

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate document format is one of the allowed formats."""
        allowed_formats = {
            "pdf",
            "csv",
            "doc",
            "docx",
            "xls",
            "xlsx",
            "html",
            "txt",
            "md",
        }
        if v.lower() not in allowed_formats:
            error_msg = f"Format must be one of: {allowed_formats}"
            raise ValueError(error_msg)
        return v.lower()

    def to_dict(self) -> dict[str, Any]:
        """Convert to API format."""
        return {
            "document": {
                "format": self.format,
                "name": self.name,
                "source": {"bytes": self.source},
            }
        }


class ToolUseContent(BaseModel):
    """Tool use content for messages."""

    tool_use_id: str = Field(..., description="Unique tool use identifier")
    name: str = Field(..., description="Tool name")
    input: dict[str, Any] = Field(default_factory=dict, description="Tool input parameters")

    def to_dict(self) -> dict[str, Any]:
        """Convert to API format."""
        return {
            "toolUse": {
                "toolUseId": self.tool_use_id,
                "name": self.name,
                "input": self.input,
            }
        }


class ToolResultContent(BaseModel):
    """Tool result content for messages."""

    tool_use_id: str = Field(..., description="Tool use identifier this result corresponds to")
    content: list[dict[str, Any]] = Field(..., description="Tool result content")
    status: str | None = Field(None, description="Tool execution status")

    def to_dict(self) -> dict[str, Any]:
        """Convert to API format."""
        result = {
            "toolResult": {
                "toolUseId": self.tool_use_id,
                "content": self.content,
            }
        }
        if self.status:
            result["toolResult"]["status"] = self.status
        return result


class Message(BaseModel):
    """A conversation message."""

    role: MessageRole = Field(..., description="Message role")
    content: list[str | ImageContent | DocumentContent | ToolUseContent | ToolResultContent] = Field(
        default_factory=list, description="Message content items"
    )

    def add_text(self, text: str) -> "Message":
        """Add text content to the message."""
        self.content.append(text)
        return self

    def add_image(self, image_bytes: bytes, image_format: str = "png") -> "Message":
        """Add image content to the message."""
        self.content.append(ImageContent(format=image_format, source=image_bytes))
        return self

    def add_document(self, doc_bytes: bytes, name: str, doc_format: str) -> "Message":
        """Add document content to the message."""
        self.content.append(DocumentContent(format=doc_format, name=name, source=doc_bytes))
        return self

    def add_tool_use(self, tool_use_id: str, name: str, input_data: dict[str, Any]) -> "Message":
        """Add tool use content to the message."""
        self.content.append(ToolUseContent(tool_use_id=tool_use_id, name=name, input=input_data))
        return self

    def add_tool_result(
        self,
        tool_use_id: str,
        content: list[dict[str, Any]],
        status: str | None = None,
    ) -> "Message":
        """Add tool result content to the message."""
        self.content.append(ToolResultContent(tool_use_id=tool_use_id, content=content, status=status))
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert message to API format."""
        content_list = []
        for item in self.content:
            if isinstance(item, str):
                content_list.append({"text": item})
            else:
                content_list.append(item.to_dict())

        return {"role": self.role.value, "content": content_list}


class ToolConfig(BaseModel):
    """Tool configuration for function calling."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    input_schema: dict[str, Any] = Field(..., description="JSON schema for tool input")

    def to_dict(self) -> dict[str, Any]:
        """Convert to API format."""
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {"json": self.input_schema},
            }
        }


class InferenceConfig(BaseModel):
    """Inference configuration for the model."""

    max_tokens: int | None = Field(None, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float | None = Field(None, ge=0.0, le=1.0, description="Sampling temperature")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Top-p sampling parameter")
    stop_sequences: list[str] | None = Field(None, description="Stop sequences")

    def to_dict(self) -> dict[str, Any]:
        """Convert to API format."""
        config = {}
        if self.max_tokens is not None:
            config["maxTokens"] = self.max_tokens
        if self.temperature is not None:
            config["temperature"] = self.temperature
        if self.top_p is not None:
            config["topP"] = self.top_p
        if self.stop_sequences is not None:
            config["stopSequences"] = self.stop_sequences
        return config


class ConversationParams(BaseModel):
    """Parameters for conversation requests."""

    model_id: SupportedModel = Field(..., description="The supported model identifier")
    messages: list[Message] | None = Field(None, description="List of messages")
    system: list[dict[str, Any]] | None = Field(None, description="System messages")
    inference_config: InferenceConfig | None = Field(None, description="Inference configuration")
    tool_config: list[ToolConfig] | None = Field(None, description="Tool configurations")
    additional_model_request_fields: dict[str, Any] | None = Field(None, description="Additional model fields")
    additional_model_response_field_paths: list[str] | None = Field(None, description="Response field paths")


class ConversationConfig(BaseModel):
    """Configuration for conversation requests."""

    model_id: str = Field(..., description="Bedrock model identifier")
    system_prompt: str | None = Field(None, description="System prompt")
    inference_config: InferenceConfig | None = Field(None, description="Inference configuration")
    tools: list[ToolConfig] | None = Field(None, description="Available tools")
    additional_model_fields: dict[str, Any] | None = Field(None, description="Additional model-specific fields")
    response_field_paths: list[str] | None = Field(None, description="Response field paths to include")
