from typing import Any

import msgspec
from pydantic import BaseModel, Field


class IntentClassificationResponse(msgspec.Struct):
    """Response structure for intent classification using msgspec for performance."""

    intent_id: str
    confidence: float
    reasoning: str


class SlotExtractionResponse(msgspec.Struct):
    """Response structure for slot extraction using msgspec for performance."""

    extracted_slots: dict[str, Any]
    confidence: float
    reasoning: str


class IntentSlot(BaseModel):
    """Definition of an intent slot/parameter."""

    name: str = Field(..., description="Slot name")
    type: str = Field(..., description="Slot data type")
    required: bool = Field(..., description="Whether the slot is required")
    description: str = Field(..., description="Description of the slot")
    examples: list[str] = Field(default_factory=list, description="Example values")


class Intent(BaseModel):
    """Definition of an intent."""

    intent_id: str = Field(..., description="Unique intent identifier")
    description: str = Field(..., description="Intent description")
    examples: list[str] = Field(default_factory=list, description="Example user queries")
    slots: list[IntentSlot] = Field(default_factory=list, description="Intent slots/parameters")


class ClassificationResult(BaseModel):
    """Result of intent classification."""

    intent_id: str = Field(..., description="Classified intent ID or fallback intent")
    confidence: float = Field(..., description="Classification confidence (0.0-1.0)")
    extracted_slots: dict[str, Any] = Field(default_factory=dict, description="Extracted slot values")
    reasoning: str = Field(..., description="Explanation of the classification")


class IntentConfig(BaseModel):
    """Configuration for intent classifier."""

    fallback_intent_id: str = Field(default="Fallback", description="Intent ID for unmatched queries")
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence for specific intent")
    temperature: float = Field(default=0.0, description="Model temperature for classification")
    max_tokens: int | None = Field(default=None, description="Maximum tokens in response")
