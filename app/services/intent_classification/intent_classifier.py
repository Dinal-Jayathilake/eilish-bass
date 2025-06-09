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
from app.models.exceptions import (
    IntentNotFoundError,
    IntentValidationError,
    LLMCommunicationError,
    PromptGenerationError,
    ResponseParsingError,
    SlotExtractionError,
)
from app.models.intent_classifier import (
    ClassificationResult,
    Intent,
    IntentClassificationResponse,
    IntentConfig,
    SlotExtractionResponse,
)
from app.services.intent_classification import utils
from app.services.intent_classification.error_handler import (
    raise_empty_response_error,
    raise_intent_loading_error,
    raise_intent_parsing_error_from_data,
    raise_intent_parsing_error_from_decode,
    raise_intent_parsing_error_from_processing,
    raise_llm_communication_error_for_classification,
    raise_llm_communication_error_for_slots,
    raise_prompt_generation_error,
    raise_prompt_generation_error_for_slots,
    raise_slot_parsing_error,
)
from app.services.logger import logger
from app.services.prompt_manager import IntentPromptManager


class IntentClassifier:
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
            self._load_intents_from_file(intents_file)
        elif intents_data:
            self._load_intents_from_data(intents_data)
        else:
            logger.warning("No intents provided during initialization")

        logger.info(
            "IntentClassifierService initialized successfully",
            num_intents=len(self.intents),
            intent_ids=[intent.intent_id for intent in self.intents],
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

        Raises:
            PromptGenerationError: If system prompt generation fails
            LLMCommunicationError: If LLM communication fails
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
            return utils.create_no_intents_fallback(self.config.fallback_intent_id)

        conversation_messages = utils.prepare_conversation_messages(user_query, messages)

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

        except (TypeError, ValueError, AttributeError) as e:
            logger.error("Failed to generate system prompt", error=str(e))
            raise_prompt_generation_error(str(e), len(self.intents))

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

            response_text = utils.extract_response_text(response)

            if not response_text:
                logger.warning("Empty response from LLM")
                return utils.create_empty_response_fallback(self.config.fallback_intent_id)

            logger.debug("Response text extracted", response_length=len(response_text))

            try:
                result = self._parse_intent_classification_response(response_text)
            except ResponseParsingError:
                logger.warning("Failed to parse response, using fallback")
                result = utils.create_parse_error_fallback(self.config.fallback_intent_id)

            if utils.should_use_fallback(result.confidence, self.config.confidence_threshold):
                logger.info(
                    "Confidence below threshold, using fallback",
                    original_intent=result.intent_id,
                    confidence=result.confidence,
                    threshold=self.config.confidence_threshold,
                )
                result = utils.create_low_confidence_response(self.config.fallback_intent_id, result.confidence)

            total_time = time.time() - start_time
            logger.info(
                "Intent classification completed",
                intent_id=result.intent_id,
                confidence=result.confidence,
                total_time_seconds=round(total_time, 3),
                llm_time_seconds=round(llm_time, 3),
            )

        except (ConnectionError, TimeoutError, KeyError, TypeError) as e:
            total_time = time.time() - start_time
            logger.error(
                "Error during intent classification",
                error=str(e),
                total_time_seconds=round(total_time, 3),
            )
            raise_llm_communication_error_for_classification(str(e), params.model_id.value)
        else:
            return result

    def extract_slots(self, user_query: str, intent_id: str) -> SlotExtractionResponse:
        """Extract slots for a known intent using direct LLM prompt.

        Args:
            user_query: User's query to extract slots from
            intent_id: ID of the intent to extract slots for

        Returns:
            SlotExtractionResponse with extracted slots

        Raises:
            IntentNotFoundError: If intent ID is not found
            PromptGenerationError: If slot extraction prompt generation fails
            LLMCommunicationError: If LLM communication fails
            SlotExtractionError: If slot extraction process fails
        """
        logger.info(
            "Starting slot extraction",
            intent_id=intent_id,
            query_length=len(user_query),
        )
        start_time = time.time()

        intent = self._get_intent_by_id()(intent_id)

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

        system_prompt = self._generate_slot_extraction_prompt(intent, user_query)
        response_text = self._call_llm_for_slot_extraction(system_prompt, user_query, intent_id)

        try:
            result = self._parse_slot_extraction_response(response_text)
        except ResponseParsingError as e:
            raise_slot_parsing_error(intent_id, str(e), len(response_text))

        total_time = time.time() - start_time
        logger.info(
            "Slot extraction completed",
            intent_id=intent_id,
            extracted_slots=list(result.extracted_slots.keys()),
            num_extracted_slots=len(result.extracted_slots),
            confidence=result.confidence,
            total_time_seconds=round(total_time, 3),
        )

        return result

    def classify(self, user_query: str, messages: list[Message] | None = None) -> ClassificationResult:
        """Complete classification with intent and slot extraction.

        Args:
            user_query: The user's input query
            messages: Previous conversation messages for context

        Returns:
            ClassificationResult with intent and extracted slots

        Raises:
            PromptGenerationError: If prompt generation fails
            LLMCommunicationError: If LLM communication fails
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

        try:
            slot_result = self.extract_slots(user_query, intent_result.intent_id)
        except (IntentNotFoundError, PromptGenerationError, LLMCommunicationError, SlotExtractionError) as e:
            logger.warning(
                "Slot extraction failed, returning classification with empty slots",
                intent_id=intent_result.intent_id,
                error=str(e),
            )

            return ClassificationResult(
                intent_id=intent_result.intent_id,
                confidence=intent_result.confidence,
                extracted_slots={},
                reasoning=utils.format_failed_slot_reasoning(intent_result.reasoning, str(e)),
            )

        final_confidence = utils.calculate_final_confidence(intent_result.confidence, slot_result.confidence)
        result = ClassificationResult(
            intent_id=intent_result.intent_id,
            confidence=final_confidence,
            extracted_slots=slot_result.extracted_slots,
            reasoning=utils.format_combined_reasoning(intent_result.reasoning, slot_result.reasoning),
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
        return result

    def _load_intents_from_file(self, file_path: str) -> None:
        """Load intents from JSON file.

        Args:
            file_path: Path to JSON file containing intent definitions

        Raises:
            IntentLoadingError: If file doesn't exist or cannot be read
            IntentParsingError: If JSON is invalid or intent data is malformed
        """
        logger.info("Loading intents from file", file_path=file_path)
        start_time = time.time()

        try:
            with Path.open(file_path, "rb") as f:
                data = msgspec.json.decode(f.read())

            logger.debug("File read successfully", file_path=file_path, data_type=type(data).__name__)

        except FileNotFoundError as e:
            logger.error("Intents file not found", file_path=file_path, error=str(e))
            raise_intent_loading_error(file_path)
        except msgspec.DecodeError as e:
            logger.error("Invalid JSON in intents file", file_path=file_path, error=str(e))
            raise_intent_parsing_error_from_decode(file_path, str(e))

        try:
            if isinstance(data, list):
                self._load_intents_from_data(data)
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

        except IntentValidationError:
            raise
        except (TypeError, ValueError, KeyError) as e:
            logger.error("Failed to process intents from file", file_path=file_path, error=str(e))
            raise_intent_parsing_error_from_processing(file_path, str(e))

    def _load_intents_from_data(self, intents_data: list[dict]) -> None:
        """Load intents from list of dictionaries.

        Args:
            intents_data: List of intent definition dictionaries

        Raises:
            IntentParsingError: If intent data is malformed
            IntentValidationError: If intent data fails validation
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

        except (TypeError, ValueError, KeyError) as e:
            logger.error("Invalid intent data during object creation", error=str(e))
            raise_intent_parsing_error_from_data(len(intents_data), str(e))

        intents_dict = [intent.model_dump() for intent in self.intents]
        if not self.prompt_manager.validate_intent_data(intents_dict):
            error_msg = "Intent data validation failed"
            logger.error("Intent data validation failed", num_intents=len(intents_dict))
            raise IntentValidationError(
                error_msg, {"num_intents": len(intents_dict), "intent_ids": [i.intent_id for i in self.intents]}
            )

        load_time = time.time() - start_time
        logger.info(
            "Intents loaded from data successfully",
            num_intents=len(self.intents),
            load_time_seconds=round(load_time, 3),
        )

    def _get_intent_by_id(self, intent_id: str) -> Intent:
        """Get intent definition by ID.

        Args:
            intent_id: Intent identifier

        Returns:
            Intent object

        Raises:
            IntentNotFoundError: If intent ID is not found
        """
        logger.debug("Looking up intent by ID", intent_id=intent_id)

        for intent in self.intents:
            if intent.intent_id == intent_id:
                logger.debug("Intent found", intent_id=intent_id, has_slots=bool(intent.slots))
                return intent

        available_intents = [i.intent_id for i in self.intents]
        logger.warning("Intent not found", intent_id=intent_id, available_intents=available_intents)
        raise IntentNotFoundError(intent_id, available_intents)

    def _parse_intent_classification_response(self, response_text: str) -> IntentClassificationResponse:
        """Parse the LLM response for intent classification.

        Args:
            response_text: Raw response text from the LLM

        Returns:
            IntentClassificationResponse with parsed data

        Raises:
            ResponseParsingError: If response cannot be parsed properly
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

        try:
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

        except (ValueError, IndexError, AttributeError) as e:
            logger.error("Failed to parse intent classification response", error=str(e))
            raise ResponseParsingError(response_text, "intent classification") from e
        else:
            return result

    def _parse_slot_extraction_response(self, response_text: str) -> SlotExtractionResponse:
        """Parse the LLM response for slot extraction.

        Args:
            response_text: Raw response text from the LLM

        Returns:
            SlotExtractionResponse with parsed data

        Raises:
            ResponseParsingError: If response cannot be parsed properly
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
            raise ResponseParsingError(response_text, "slot extraction") from e

    def _generate_slot_extraction_prompt(self, intent: Intent, user_query: str) -> str:
        """Generate slot extraction prompt.

        Args:
            intent: Intent object
            user_query: User query

        Returns:
            Generated prompt string

        Raises:
            PromptGenerationError: If prompt generation fails
        """
        try:
            intent_data = intent.model_dump()
            system_prompt = self.prompt_manager.render_slot_extraction_prompt(
                intent_definition=intent_data, user_query=user_query
            )
            logger.debug("Slot extraction prompt generated", prompt_length=len(system_prompt))
        except (TypeError, ValueError, AttributeError) as e:
            logger.error("Failed to generate slot extraction prompt", intent_id=intent.intent_id, error=str(e))
            raise_prompt_generation_error_for_slots(intent.intent_id, str(e))
        else:
            return system_prompt

    def _call_llm_for_slot_extraction(self, system_prompt: str, user_query: str, intent_id: str) -> str:
        """Call LLM for slot extraction.

        Args:
            system_prompt: System prompt for LLM
            user_query: User query
            intent_id: Intent identifier for logging

        Returns:
            Response text from LLM

        Raises:
            SlotExtractionError: If LLM returns empty response
            LLMCommunicationError: If LLM communication fails
        """
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

            response_text = utils.extract_response_text(response)

            if not response_text:
                logger.warning("Empty response from slot extraction LLM", intent_id=intent_id)
                raise_empty_response_error(intent_id)

        except SlotExtractionError:
            raise
        except (ConnectionError, TimeoutError, KeyError, TypeError) as e:
            raise_llm_communication_error_for_slots(intent_id, str(e))
        else:
            return response_text
