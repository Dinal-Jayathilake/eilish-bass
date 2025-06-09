import json
import time

from app.services.intent_classification.intent_classifier import create_classifier_from_file
from app.services.logger import logger

try:
    classifier = create_classifier_from_file(
        intents_file="app/services/intent_classification/intent_configs/csr_intents.json",
        fallback_intent_id="Fallback",
        confidence_threshold=0.8,
    )
except (FileNotFoundError, ValueError, KeyError, TypeError):
    logger.exception("Failed to initialize classifier")
    classifier = None


def lambda_handler(event: dict, _context: object) -> dict:
    """Simple Lambda handler for intent classification testing."""
    if classifier is None:
        return {"statusCode": 500, "body": json.dumps({"error": "Service unavailable"})}
    try:
        body = json.loads(event["body"]) if isinstance(event.get("body"), str) else event.get("body", {})

        query = body.get("query")
        if not query:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing query in request body"})}

        result = classifier.classify(query)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "intent_id": result.intent_id,
                    "confidence": result.confidence,
                    "extracted_slots": result.extracted_slots,
                    "reasoning": result.reasoning,
                }
            ),
        }

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.exception("Exception during lambda_handler")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


if __name__ == "__main__":
    test_queries = [
        # "Can you show me a summary of customer 123456789?",
        # "What's my balance?",
        "Transfer $1000 to account 987654321",
        # "Hello there!",
        # "Random gibberish that won't match any intent",
    ]

    for query in test_queries:
        logger.info("Testing query: '%s'", query)
        test_event = {"body": {"query": query}}
        response = lambda_handler(test_event, {})
        logger.info("Response: %s", json.loads(response["body"]), indent=2)
        time.sleep(1.5)
