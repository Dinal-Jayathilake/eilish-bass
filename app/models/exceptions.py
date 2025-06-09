RESPONSE_PREVIEW_LENGTH = 200


class IntentClassificationError(Exception):
    """Base exception for all intent classification related errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Human readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            return f"{self.message}. Details: {self.details}"
        return self.message


class IntentLoadingError(IntentClassificationError):
    """Raised when there's an error loading intent definitions."""


class IntentValidationError(IntentClassificationError):
    """Raised when intent data fails validation."""


class IntentParsingError(IntentClassificationError):
    """Raised when there's an error parsing intent data from file or JSON."""


class PromptGenerationError(IntentClassificationError):
    """Raised when there's an error generating prompts for LLM."""


class LLMCommunicationError(IntentClassificationError):
    """Raised when there's an error communicating with the LLM."""


class SlotExtractionError(IntentClassificationError):
    """Raised when there's an error during slot extraction."""


class IntentNotFoundError(IntentClassificationError):
    """Raised when a requested intent ID is not found."""

    def __init__(self, intent_id: str, available_intents: list[str] | None = None) -> None:
        """Initialize the exception.

        Args:
            intent_id: The intent ID that was not found
            available_intents: List of available intent IDs for context
        """
        message = f"Intent '{intent_id}' not found"
        details = {"intent_id": intent_id}
        if available_intents:
            details["available_intents"] = available_intents
        super().__init__(message, details)
        self.intent_id = intent_id
        self.available_intents = available_intents or []


class ResponseParsingError(IntentClassificationError):
    """Raised when there's an error parsing LLM responses."""

    def __init__(self, response_text: str, parsing_context: str | None = None) -> None:
        """Initialize the exception.

        Args:
            response_text: The response text that failed to parse
            parsing_context: Additional context about what was being parsed
        """
        message = f"Failed to parse LLM response{f' for {parsing_context}' if parsing_context else ''}"
        details = {
            "response_length": len(response_text),
            "response_preview": response_text[:RESPONSE_PREVIEW_LENGTH] + "..."
            if len(response_text) > RESPONSE_PREVIEW_LENGTH
            else response_text,
        }
        if parsing_context:
            details["parsing_context"] = parsing_context
        super().__init__(message, details)
        self.response_text = response_text
        self.parsing_context = parsing_context
