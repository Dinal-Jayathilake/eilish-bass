from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

from app.services.logger import logger


class PromptManager:
    """Generic prompt manager for handling Jinja2 templates with caching."""

    def __init__(self, templates_dir: str | Path | None = None) -> None:
        """Initialize the prompt manager.

        Args:
            templates_dir: Directory containing Jinja2 templates.
                          If None, will be set relative to caller.
        """
        if templates_dir is None:
            templates_dir = Path.cwd() / "app" / "services" / "prompt_templates"

        self.templates_dir = Path(templates_dir)

        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
            cache_size=200,
            auto_reload=False,
            enable_async=False,
        )

        self._template_cache: dict[str, Template] = {}

        logger.info("PromptManager initialized", templates_dir=str(self.templates_dir))

    def get_template(self, template_name: str, *, use_cache: bool = True) -> Template:
        """Get a template by name with optional caching.

        Args:
            template_name: Name of the template file
            use_cache: Whether to use internal template cache

        Returns:
            Compiled Jinja2 template
        """
        if use_cache and template_name in self._template_cache:
            logger.debug("Template retrieved from cache", template_name=template_name)
            return self._template_cache[template_name]

        logger.debug("Loading template from environment", template_name=template_name)
        template = self.env.get_template(template_name)

        if use_cache:
            self._template_cache[template_name] = template
            logger.debug("Template cached", template_name=template_name)

        return template

    def render_template(self, template_name: str, **kwargs: dict[str, Any]) -> str:
        """Render a template with the given variables.

        Args:
            template_name: Name of the template file
            **kwargs: Variables to pass to the template

        Returns:
            Rendered template string
        """
        logger.debug("Rendering template", template_name=template_name, variables=kwargs.keys())
        template = self.get_template(template_name)
        rendered = template.render(**kwargs)
        logger.debug("Template rendered", template_name=template_name)
        return rendered

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._template_cache.clear()
        self.env.cache.clear()
        logger.info("Template cache cleared")


class IntentPromptManager(PromptManager):
    """Specialized prompt manager for intent classification templates."""

    def __init__(self, templates_dir: str | Path | None = None) -> None:
        """Initialize intent-specific prompt manager."""
        if templates_dir is None:
            templates_dir = Path(__file__).parent.parent / "services" / "intent_classification" / "prompt_templates"

        super().__init__(templates_dir)

        self._intent_classification_template: Template | None = None
        self._slot_extraction_template: Template | None = None

        logger.info("IntentPromptManager initialized", templates_dir=str(self.templates_dir))

    @property
    def intent_classification_template(self) -> Template:
        """Get the intent classification template (cached)."""
        if self._intent_classification_template is None:
            logger.debug("Loading intent classification template")
            self._intent_classification_template = self.get_template("intent_classification.j2")
        return self._intent_classification_template

    @property
    def slot_extraction_template(self) -> Template:
        """Get the slot extraction template (cached)."""
        if self._slot_extraction_template is None:
            logger.debug("Loading slot extraction template")
            self._slot_extraction_template = self.get_template("slot_extraction.j2")
        return self._slot_extraction_template

    def render_intent_classification_prompt(
        self,
        intents: list[dict[str, Any]],
        fallback_intent_id: str = "Fallback",
        confidence_threshold: float = 0.7,
        **additional_vars: dict[str, Any],
    ) -> str:
        """Render the intent classification system prompt.

        Args:
            intents: List of intent definitions
            fallback_intent_id: Intent ID to use for fallback
            confidence_threshold: Minimum confidence threshold
            **additional_vars: Additional variables for the template

        Returns:
            Rendered system prompt
        """
        logger.debug(
            "Rendering intent classification prompt",
            fallback_intent_id=fallback_intent_id,
            confidence_threshold=confidence_threshold,
        )
        return self.intent_classification_template.render(
            intents=intents,
            fallback_intent_id=fallback_intent_id,
            confidence_threshold=confidence_threshold,
            **additional_vars,
        )

    def render_slot_extraction_prompt(
        self, intent_definition: dict[str, Any], user_query: str, **additional_vars: dict[str, Any]
    ) -> str:
        """Render the slot extraction system prompt.

        Args:
            intent_definition: Definition of the intent with slots
            user_query: User's query to extract slots from
            **additional_vars: Additional variables for the template

        Returns:
            Rendered system prompt
        """
        logger.debug("Rendering slot extraction prompt", intent_id=intent_definition.get("intent_id"))
        return self.slot_extraction_template.render(intent=intent_definition, user_query=user_query, **additional_vars)

    def validate_intent_data(self, intents: list[dict[str, Any]]) -> bool:
        """Validate that intent data has required fields.

        Args:
            intents: List of intent definitions

        Returns:
            True if valid, False otherwise
        """
        logger.debug("Validating intent data", num_intents=len(intents))

        required_fields = ["intent_id", "description"]

        for intent in intents:
            if not all(field in intent for field in required_fields):
                logger.warning("Intent missing required fields", intent_id=intent.get("intent_id"))
                return False

            if "slots" in intent:
                slot_required_fields = ["name", "type", "required", "description"]
                for slot in intent["slots"]:
                    if not all(field in slot for field in slot_required_fields):
                        logger.warning(
                            "Slot missing required fields", intent_id=intent.get("intent_id"), slot=slot.get("name")
                        )
                        return False

        logger.debug("Intent data validation passed")
        return True
