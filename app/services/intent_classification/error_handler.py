from app.models.exceptions import (
    IntentLoadingError,
    IntentParsingError,
    LLMCommunicationError,
    PromptGenerationError,
    SlotExtractionError,
)


def raise_intent_loading_error(file_path: str) -> None:
    """Raise intent loading error with context.

    Args:
        file_path: Path that failed to load
    """
    error_msg = f"Intents file not found: {file_path}"
    raise IntentLoadingError(error_msg, {"file_path": file_path})


def raise_intent_parsing_error_from_decode(file_path: str, decode_error: str) -> None:
    """Raise intent parsing error from decode failure.

    Args:
        file_path: Path that failed to decode
        decode_error: Decode error message
    """
    error_msg = f"Invalid JSON in intents file: {decode_error}"
    raise IntentParsingError(error_msg, {"file_path": file_path, "decode_error": decode_error})


def raise_intent_parsing_error_from_processing(file_path: str, error: str) -> None:
    """Raise intent parsing error from processing failure.

    Args:
        file_path: Path that failed to process
        error: Processing error message
    """
    error_msg = f"Failed to process intents from file: {error}"
    raise IntentParsingError(error_msg, {"file_path": file_path, "error": error})


def raise_intent_parsing_error_from_data(num_intents: int, error: str) -> None:
    """Raise intent parsing error from data processing.

    Args:
        num_intents: Number of intents being processed
        error: Error message
    """
    error_msg = f"Invalid intent data: {error}"
    raise IntentParsingError(error_msg, {"num_intents": num_intents, "error": error})


def raise_prompt_generation_error(error: str, context: dict) -> None:
    """Raise prompt generation error with context.

    Args:
        error: Error message
        context: Additional error context
    """
    error_msg = f"Failed to generate system prompt: {error}"
    raise PromptGenerationError(error_msg, context)


def raise_llm_communication_error_for_classification(error: str, model_id: str) -> None:
    """Raise LLM communication error for classification.

    Args:
        error: Error message
        model_id: Model identifier
    """
    error_msg = f"Error during intent classification: {error}"
    raise LLMCommunicationError(error_msg, {"error": error, "model_id": model_id})


def raise_prompt_generation_error_for_slots(intent_id: str, error: str) -> None:
    """Raise prompt generation error for slot extraction.

    Args:
        intent_id: Intent identifier
        error: Error message
    """
    error_msg = f"Failed to generate slot extraction prompt: {error}"
    raise PromptGenerationError(error_msg, {"intent_id": intent_id, "error": error})


def raise_empty_response_error(intent_id: str) -> None:
    """Raise error for empty LLM response.

    Args:
        intent_id: Intent identifier
    """
    error_msg = "Empty response from model during slot extraction"
    raise SlotExtractionError(error_msg, {"intent_id": intent_id})


def raise_llm_communication_error_for_slots(intent_id: str, error: str) -> None:
    """Raise LLM communication error for slot extraction.

    Args:
        intent_id: Intent identifier
        error: Error message
    """
    error_msg = f"Error during slot extraction: {error}"
    raise LLMCommunicationError(error_msg, {"intent_id": intent_id, "error": error})


def raise_slot_parsing_error(intent_id: str, error: str, response_length: int) -> None:
    """Raise slot parsing error.

    Args:
        intent_id: Intent identifier
        error: Error message
        response_length: Length of response that failed to parse
    """
    error_msg = f"Failed to parse slot extraction response: {error}"
    raise SlotExtractionError(error_msg, {"intent_id": intent_id, "response_length": response_length})
