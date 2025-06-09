import time
from pathlib import Path

import msgspec

from app.clients.bedrock_chat import BedrockChatClient
from app.models.bedrock_chat import (
    ConversationParams,
    InferenceConfig,
    Message,
    MessageRole,
    SupportedModel,
)
from app.models.intent_classifier import (
    ClassificationResult,
    Intent,
    IntentClassificationResponse,
    IntentConfig,
    SlotExtractionResponse,
)
from app.services.logger import logger
from app.services.prompt_manager import IntentPromptManager


class IntentClassifierService:
    """High-performance intent classification service using direct LLM prompts."""

    def __init__(
        self,
        intents_file: str | None = None,
        intents_data: list[dict] | None = None,
        config: IntentConfig | None = None,
        prompt_manager: IntentPromptManager | None = None,
    ) -> None:
        """Initialize the intent classifier service.

        Args:
            intents_file: Path to JSON file containing intents
            intents_data: List of intent dictionaries (alternative to file)
            config: Configuration for the classifier
            prompt_manager: Optional custom prompt manager instance
        """
        logger.info(
            "Initializing IntentClassifierService",
            has_intents_file=intents_file is not None,
            has_intents_data=intents_data is not None,
            has_custom_config=config is not None,
            has_custom_prompt_manager=prompt_manager is not None,
        )

        self.client = BedrockChatClient()
        self.config = config or IntentConfig()
        self.intents: list[Intent] = []
        self.prompt_manager = prompt_manager or IntentPromptManager()

        logger.debug(
            "Service configuration",
            fallback_intent_id=self.config.fallback_intent_id,
            confidence_threshold=self.config.confidence_threshold,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        if intents_file:
            self.load_intents_from_file(intents_file)
        elif intents_data:
            self.load_intents_from_data(intents_data)
        else:
            logger.warning("No intents provided during initialization")

        logger.info(
            "IntentClassifierService initialized successfully",
            num_intents=len(self.intents),
            intent_ids=[intent.intent_id for intent in self.intents],
        )

    def load_intents_from_file(self, file_path: str) -> None:
        """Load intents from JSON file.

        Args:
            file_path: Path to JSON file containing intent definitions

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid or intent data is malformed
        """
        logger.info("Loading intents from file", file_path=file_path)
        start_time = time.time()

        try:
            with Path.open(file_path, "rb") as f:
                data = msgspec.json.decode(f.read())

            logger.debug("File read successfully", file_path=file_path, data_type=type(data).__name__)

        except FileNotFoundError as e:
            error_msg = f"Intents file not found: {file_path}"
            logger.error("Intents file not found", file_path=file_path, error=str(e))
            raise FileNotFoundError(error_msg) from e
        except msgspec.DecodeError as e:
            error_msg = f"Invalid JSON in intents file: {e}"
            logger.error("Invalid JSON in intents file", file_path=file_path, error=str(e))
            raise ValueError(error_msg) from e

        try:
            if isinstance(data, list):
                self.load_intents_from_data(data)
            else:
                self.intents = [Intent(**data)]
                logger.debug("Loaded single intent from file", intent_id=self.intents[0].intent_id)

            load_time = time.time() - start_time
            logger.info(
                "Intents loaded from file successfully",
                file_path=file_path,
                num_intents=len(self.intents),
                load_time_seconds=round(load_time, 3),
            )

        except Exception as e:
            logger.error("Failed to process intents from file", file_path=file_path, error=str(e))
            raise

    def load_intents_from_data(self, intents_data: list[dict]) -> None:
        """Load intents from list of dictionaries.

        Args:
            intents_data: List of intent definition dictionaries

        Raises:
            ValueError: If intent data is malformed
        """
        logger.info("Loading intents from data", num_intents=len(intents_data))
        start_time = time.time()

        try:
            self.intents = [Intent(**intent_data) for intent_data in intents_data]
            logger.debug(
                "Intent objects created",
                intent_ids=[intent.intent_id for intent in self.intents],
                total_slots=sum(len(intent.slots or []) for intent in self.intents),
            )

        except (ValueError, KeyError, TypeError) as e:
            error_msg = f"Invalid intent data: {e}"
            logger.error("Invalid intent data during object creation", error=str(e))
            raise ValueError(error_msg) from e

        intents_dict = [intent.model_dump() for intent in self.intents]
        if not self.prompt_manager.validate_intent_data(intents_dict):
            error_msg = "Intent data validation failed"
            logger.error("Intent data validation failed", num_intents=len(intents_dict))
            raise ValueError(error_msg)

        load_time = time.time() - start_time
        logger.info(
            "Intents loaded from data successfully",
            num_intents=len(self.intents),
            load_time_seconds=round(load_time, 3),
        )

    def get_intent_by_id(self, intent_id: str) -> Intent | None:
        """Get intent definition by ID.

        Args:
            intent_id: Intent identifier

        Returns:
            Intent object if found, None otherwise
        """
        logger.debug("Looking up intent by ID", intent_id=intent_id)

        for intent in self.intents:
            if intent.intent_id == intent_id:
                logger.debug("Intent found", intent_id=intent_id, has_slots=bool(intent.slots))
                return intent

        logger.warning("Intent not found", intent_id=intent_id, available_intents=[i.intent_id for i in self.intents])
        return None

    def _parse_intent_classification_response(self, response_text: str) -> IntentClassificationResponse:
        """Parse the LLM response for intent classification.

        Args:
            response_text: Raw response text from the LLM

        Returns:
            IntentClassificationResponse with parsed data
        """
        logger.debug("Parsing intent classification response", response_length=len(response_text))

        try:
            if response_text.strip().startswith("{"):
                data = msgspec.json.decode(response_text.strip())
                result = IntentClassificationResponse(
                    intent_id=data.get("intent_id", self.config.fallback_intent_id),
                    confidence=float(data.get("confidence", 0.0)),
                    reasoning=data.get("reasoning", "No reasoning provided"),
                )
                logger.debug(
                    "Successfully parsed JSON response",
                    intent_id=result.intent_id,
                    confidence=result.confidence,
                )
                return result
        except (msgspec.DecodeError, ValueError, KeyError) as e:
            logger.debug("JSON parsing failed, trying text parsing", error=str(e))

        lines = response_text.strip().split("\n")
        intent_id = self.config.fallback_intent_id
        confidence = 0.0
        reasoning = "Could not parse response"

        for raw_line in lines:
            line = raw_line.strip()
            if line.lower().startswith("intent:") or line.lower().startswith("intent_id:"):
                intent_id = line.split(":", 1)[1].strip().strip("\"'")
            elif line.lower().startswith("confidence:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    confidence = 0.0
            elif line.lower().startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip().strip("\"'")

        result = IntentClassificationResponse(
            intent_id=intent_id,
            confidence=confidence,
            reasoning=reasoning,
        )

        logger.debug(
            "Parsed text response",
            intent_id=result.intent_id,
            confidence=result.confidence,
            parsing_method="text",
        )

        return result

    def _parse_slot_extraction_response(self, response_text: str) -> SlotExtractionResponse:
        """Parse the LLM response for slot extraction.

        Args:
            response_text: Raw response text from the LLM

        Returns:
            SlotExtractionResponse with parsed data
        """
        logger.debug("Parsing slot extraction response", response_length=len(response_text))

        try:
            if response_text.strip().startswith("{"):
                data = msgspec.json.decode(response_text.strip())
                result = SlotExtractionResponse(
                    extracted_slots=data.get("extracted_slots", {}),
                    confidence=float(data.get("confidence", 0.0)),
                    reasoning=data.get("reasoning", "No reasoning provided"),
                )
                logger.debug(
                    "Successfully parsed slot extraction JSON",
                    extracted_slots=list(result.extracted_slots.keys()),
                    num_slots=len(result.extracted_slots),
                    confidence=result.confidence,
                )
                return result
        except (msgspec.DecodeError, ValueError, KeyError) as e:
            logger.debug("Slot extraction JSON parsing failed", error=str(e))

        logger.warning("Could not parse slot extraction response, returning empty result")
        return SlotExtractionResponse(
            extracted_slots={},
            confidence=0.0,
            reasoning="Could not parse slot extraction response",
        )

    def classify_intent_only(
        self, user_query: str, messages: list[Message] | None = None
    ) -> IntentClassificationResponse:
        """Classify user intent without slot extraction using direct LLM prompt.

        Args:
            user_query: The user's input query
            messages: Previous conversation messages for context

        Returns:
            IntentClassificationResponse with intent classification only
        """
        logger.info(
            "Starting intent classification",
            query_length=len(user_query),
            has_context_messages=bool(messages),
            num_context_messages=len(messages) if messages else 0,
        )
        start_time = time.time()

        if not self.intents:
            logger.error("No intents loaded for classification")
            return IntentClassificationResponse(
                intent_id=self.config.fallback_intent_id,
                confidence=0.0,
                reasoning="No intents loaded",
            )

        messages = messages or []
        conversation_messages = messages.copy()
        conversation_messages.append(Message(role=MessageRole.USER).add_text(user_query))

        logger.debug(
            "Prepared conversation messages",
            total_messages=len(conversation_messages),
            final_message_role=conversation_messages[-1].role.value,
        )

        try:
            intents_data = [intent.model_dump() for intent in self.intents]
            system_prompt = self.prompt_manager.render_intent_classification_prompt(
                intents=intents_data,
                fallback_intent_id=self.config.fallback_intent_id,
                confidence_threshold=self.config.confidence_threshold,
            )
            logger.debug("System prompt generated", prompt_length=len(system_prompt))

        except (ValueError, KeyError, TypeError) as e:
            logger.error("Failed to generate system prompt", error=str(e))
            return IntentClassificationResponse(
                intent_id=self.config.fallback_intent_id,
                confidence=0.0,
                reasoning=f"Failed to generate system prompt: {e}",
            )

        params = ConversationParams(
            model_id=SupportedModel.CLAUDE_3_HAIKU,
            messages=conversation_messages,
            system=[{"text": system_prompt}],
            inference_config=InferenceConfig(temperature=self.config.temperature, max_tokens=self.config.max_tokens),
        )

        logger.debug(
            "Calling LLM for intent classification",
            model_id=params.model_id.value,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        try:
            llm_start_time = time.time()
            response = self.client.converse(params)
            llm_time = time.time() - llm_start_time

            logger.debug("LLM response received", llm_time_seconds=round(llm_time, 3))

            response_text = ""
            if "output" in response and "message" in response["output"]:
                content = response["output"]["message"]["content"]
                for item in content:
                    if "text" in item:
                        response_text += item["text"]

            if not response_text:
                logger.warning("Empty response from LLM")
                return IntentClassificationResponse(
                    intent_id=self.config.fallback_intent_id,
                    confidence=0.0,
                    reasoning="Empty response from model",
                )

            logger.debug("Response text extracted", response_length=len(response_text))

            result = self._parse_intent_classification_response(response_text)

            if result.confidence < self.config.confidence_threshold:
                logger.info(
                    "Confidence below threshold, using fallback",
                    original_intent=result.intent_id,
                    confidence=result.confidence,
                    threshold=self.config.confidence_threshold,
                )
                result = IntentClassificationResponse(
                    intent_id=self.config.fallback_intent_id,
                    confidence=result.confidence,
                    reasoning=f"Low confidence ({result.confidence:.2f}) for any specific intent. Using fallback.",
                )

            total_time = time.time() - start_time
            logger.info(
                "Intent classification completed",
                intent_id=result.intent_id,
                confidence=result.confidence,
                total_time_seconds=round(total_time, 3),
                llm_time_seconds=round(llm_time, 3),
            )

        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            total_time = time.time() - start_time
            logger.error(
                "Error during intent classification",
                error=str(e),
                total_time_seconds=round(total_time, 3),
            )
            return IntentClassificationResponse(
                intent_id=self.config.fallback_intent_id,
                confidence=0.0,
                reasoning=f"Error during classification: {e}",
            )
        else:
            return result

    def extract_slots(self, user_query: str, intent_id: str) -> SlotExtractionResponse:
        """Extract slots for a known intent using direct LLM prompt.

        Args:
            user_query: User's query to extract slots from
            intent_id: ID of the intent to extract slots for

        Returns:
            SlotExtractionResponse with extracted slots
        """
        logger.info(
            "Starting slot extraction",
            intent_id=intent_id,
            query_length=len(user_query),
        )
        start_time = time.time()

        intent = self.get_intent_by_id(intent_id)
        if not intent:
            logger.error("Intent not found for slot extraction", intent_id=intent_id)
            return SlotExtractionResponse(extracted_slots={}, confidence=0.0, reasoning=f"Intent {intent_id} not found")

        if not intent.slots:
            logger.info("Intent has no slots to extract", intent_id=intent_id)
            return SlotExtractionResponse(
                extracted_slots={}, confidence=1.0, reasoning="Intent has no slots to extract"
            )

        logger.debug(
            "Intent found with slots",
            intent_id=intent_id,
            num_slots=len(intent.slots),
            slot_names=[slot.name for slot in intent.slots],
        )

        try:
            intent_data = intent.model_dump()
            system_prompt = self.prompt_manager.render_slot_extraction_prompt(
                intent_definition=intent_data, user_query=user_query
            )
            logger.debug("Slot extraction prompt generated", prompt_length=len(system_prompt))

        except (ValueError, KeyError, TypeError) as e:
            logger.error("Failed to generate slot extraction prompt", intent_id=intent_id, error=str(e))
            return SlotExtractionResponse(
                extracted_slots={},
                confidence=0.0,
                reasoning=f"Failed to generate slot extraction prompt: {e}",
            )

        params = ConversationParams(
            model_id=SupportedModel.CLAUDE_3_HAIKU,
            messages=[Message(role=MessageRole.USER).add_text(user_query)],
            system=[{"text": system_prompt}],
            inference_config=InferenceConfig(temperature=self.config.temperature, max_tokens=self.config.max_tokens),
        )

        logger.debug("Calling LLM for slot extraction", intent_id=intent_id)

        try:
            llm_start_time = time.time()
            response = self.client.converse(params)
            llm_time = time.time() - llm_start_time

            logger.debug("Slot extraction LLM response received", llm_time_seconds=round(llm_time, 3))

            response_text = ""
            if "output" in response and "message" in response["output"]:
                content = response["output"]["message"]["content"]
                for item in content:
                    if "text" in item:
                        response_text += item["text"]

            if not response_text:
                logger.warning("Empty response from slot extraction LLM", intent_id=intent_id)
                return SlotExtractionResponse(
                    extracted_slots={},
                    confidence=0.0,
                    reasoning="Empty response from model",
                )

            result = self._parse_slot_extraction_response(response_text)

            total_time = time.time() - start_time
            logger.info(
                "Slot extraction completed",
                intent_id=intent_id,
                extracted_slots=list(result.extracted_slots.keys()),
                num_extracted_slots=len(result.extracted_slots),
                confidence=result.confidence,
                total_time_seconds=round(total_time, 3),
                llm_time_seconds=round(llm_time, 3),
            )

        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            total_time = time.time() - start_time
            logger.error(
                "Error during slot extraction",
                intent_id=intent_id,
                error=str(e),
                total_time_seconds=round(total_time, 3),
            )
            return SlotExtractionResponse(
                extracted_slots={}, confidence=0.0, reasoning=f"Error during slot extraction: {e}"
            )
        else:
            return result

    def classify(self, user_query: str, messages: list[Message] | None = None) -> ClassificationResult:
        """Complete classification with intent and slot extraction.

        Args:
            user_query: The user's input query
            messages: Previous conversation messages for context

        Returns:
            ClassificationResult with intent and extracted slots
        """
        logger.info(
            "Starting complete classification",
            query_length=len(user_query),
            has_context_messages=bool(messages),
        )
        start_time = time.time()

        intent_result = self.classify_intent_only(user_query, messages)

        if intent_result.intent_id == self.config.fallback_intent_id:
            total_time = time.time() - start_time
            logger.info(
                "Classification completed with fallback intent",
                intent_id=intent_result.intent_id,
                confidence=intent_result.confidence,
                total_time_seconds=round(total_time, 3),
            )
            return ClassificationResult(
                intent_id=intent_result.intent_id,
                confidence=intent_result.confidence,
                extracted_slots={},
                reasoning=intent_result.reasoning,
            )

        slot_result = self.extract_slots(user_query, intent_result.intent_id)

        final_confidence = min(intent_result.confidence, slot_result.confidence)
        result = ClassificationResult(
            intent_id=intent_result.intent_id,
            confidence=final_confidence,
            extracted_slots=slot_result.extracted_slots,
            reasoning=f"Intent: {intent_result.reasoning}. Slots: {slot_result.reasoning}",
        )

        total_time = time.time() - start_time
        logger.info(
            "Complete classification finished",
            intent_id=result.intent_id,
            final_confidence=final_confidence,
            intent_confidence=intent_result.confidence,
            slot_confidence=slot_result.confidence,
            num_extracted_slots=len(result.extracted_slots),
            total_time_seconds=round(total_time, 3),
        )

        assert isinstance(result, ClassificationResult), "Unexpected result type in classify()"
        return result


def create_classifier(
    intents_data: list[dict],
    fallback_intent_id: str = "Fallback",
    confidence_threshold: float = 0.8,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> IntentClassifierService:
    """Factory function to create a classifier with common settings.

    Args:
        intents_data: List of intent definition dictionaries
        fallback_intent_id: Intent ID for unmatched queries
        confidence_threshold: Minimum confidence for specific intents
        temperature: Model temperature for classification
        max_tokens: Maximum tokens in model response

    Returns:
        Configured IntentClassifierService instance
    """
    logger.info(
        "Creating classifier from data",
        num_intents=len(intents_data),
        fallback_intent_id=fallback_intent_id,
        confidence_threshold=confidence_threshold,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    config = IntentConfig(
        fallback_intent_id=fallback_intent_id,
        confidence_threshold=confidence_threshold,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    classifier = IntentClassifierService(intents_data=intents_data, config=config)
    logger.info("Classifier created successfully from data")
    return classifier


def create_classifier_from_file(
    intents_file: str,
    fallback_intent_id: str = "Fallback",
    confidence_threshold: float = 0.8,
    temperature: float = 0,
    max_tokens: int | None = None,
) -> IntentClassifierService:
    """Factory function to create a classifier from intents file.

    Args:
        intents_file: Path to JSON file containing intent definitions
        fallback_intent_id: Intent ID for unmatched queries
        confidence_threshold: Minimum confidence for specific intents
        temperature: Model temperature for classification
        max_tokens: Maximum tokens in model response

    Returns:
        Configured IntentClassifierService instance
    """
    logger.info(
        "Creating classifier from file",
        intents_file=intents_file,
        fallback_intent_id=fallback_intent_id,
        confidence_threshold=confidence_threshold,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    config = IntentConfig(
        fallback_intent_id=fallback_intent_id,
        confidence_threshold=confidence_threshold,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    classifier = IntentClassifierService(intents_file=intents_file, config=config)
    logger.info("Classifier created successfully from file")
    return classifier
