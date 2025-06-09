from app.models.bedrock_chat import Message, MessageRole
from app.models.intent_classifier import IntentClassificationResponse


def prepare_conversation_messages(user_query: str, messages: list[Message] | None = None) -> list[Message]:
    """Prepare conversation messages for classification.

    Args:
        user_query: User's input query
        messages: Previous conversation messages

    Returns:
        List of conversation messages with user query appended
    """
    messages = messages or []
    conversation_messages = messages.copy()
    conversation_messages.append(Message(role=MessageRole.USER).add_text(user_query))
    return conversation_messages


def create_fallback_response(
    fallback_intent_id: str, confidence: float = 0.0, reason: str = "Fallback response"
) -> IntentClassificationResponse:
    """Create a fallback intent classification response.

    Args:
        fallback_intent_id: ID of the fallback intent
        confidence: Confidence score for the response
        reason: Reason for the fallback

    Returns:
        IntentClassificationResponse with fallback values
    """
    return IntentClassificationResponse(
        intent_id=fallback_intent_id,
        confidence=confidence,
        reasoning=reason,
    )


def create_empty_response_fallback(fallback_intent_id: str) -> IntentClassificationResponse:
    """Create fallback response for empty LLM responses.

    Args:
        fallback_intent_id: ID of the fallback intent

    Returns:
        IntentClassificationResponse for empty response scenario
    """
    return IntentClassificationResponse(
        intent_id=fallback_intent_id,
        confidence=0.0,
        reasoning="Empty response from model",
    )


def create_no_intents_fallback(fallback_intent_id: str) -> IntentClassificationResponse:
    """Create fallback response when no intents are loaded.

    Args:
        fallback_intent_id: ID of the fallback intent

    Returns:
        IntentClassificationResponse for no intents scenario
    """
    return IntentClassificationResponse(
        intent_id=fallback_intent_id,
        confidence=0.0,
        reasoning="No intents loaded",
    )


def create_parse_error_fallback(fallback_intent_id: str) -> IntentClassificationResponse:
    """Create fallback response for parsing errors.

    Args:
        fallback_intent_id: ID of the fallback intent

    Returns:
        IntentClassificationResponse for parse error scenario
    """
    return IntentClassificationResponse(
        intent_id=fallback_intent_id,
        confidence=0.0,
        reasoning="Failed to parse model response",
    )


def should_use_fallback(confidence: float, threshold: float) -> bool:
    """Check if confidence is below threshold requiring fallback.

    Args:
        confidence: Confidence score to check
        threshold: Minimum confidence threshold

    Returns:
        True if should use fallback, False otherwise
    """
    return confidence < threshold


def create_low_confidence_response(fallback_intent_id: str, original_confidence: float) -> IntentClassificationResponse:
    """Create response for low confidence scenarios.

    Args:
        fallback_intent_id: ID of the fallback intent
        original_confidence: Original confidence score

    Returns:
        IntentClassificationResponse with low confidence reasoning
    """
    return IntentClassificationResponse(
        intent_id=fallback_intent_id,
        confidence=original_confidence,
        reasoning=f"Low confidence ({original_confidence:.2f}) for any specific intent. Using fallback.",
    )


def extract_response_text(response: dict) -> str:
    """Extract text content from LLM response.

    Args:
        response: Raw response from LLM

    Returns:
        Extracted text content, empty string if none found
    """
    response_text = ""
    if "output" in response and "message" in response["output"]:
        content = response["output"]["message"]["content"]
        for item in content:
            if "text" in item:
                response_text += item["text"]
    return response_text


def calculate_final_confidence(intent_confidence: float, slot_confidence: float) -> float:
    """Calculate final confidence from intent and slot confidences.

    Args:
        intent_confidence: Confidence from intent classification
        slot_confidence: Confidence from slot extraction

    Returns:
        Final confidence score (minimum of both)
    """
    return min(intent_confidence, slot_confidence)


def format_combined_reasoning(intent_reasoning: str, slot_reasoning: str) -> str:
    """Format combined reasoning from intent and slot extraction.

    Args:
        intent_reasoning: Reasoning from intent classification
        slot_reasoning: Reasoning from slot extraction

    Returns:
        Combined reasoning string
    """
    return f"Intent: {intent_reasoning}. Slots: {slot_reasoning}"


def format_failed_slot_reasoning(intent_reasoning: str, error: str) -> str:
    """Format reasoning when slot extraction fails.

    Args:
        intent_reasoning: Reasoning from intent classification
        error: Error from slot extraction

    Returns:
        Combined reasoning with error information
    """
    return f"Intent: {intent_reasoning}. Slot extraction failed: {error}"
