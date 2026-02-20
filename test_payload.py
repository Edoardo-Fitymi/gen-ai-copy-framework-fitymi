"""
Unit tests for FitymiPayload and TopologicConstraints dataclasses.
"""
import pytest
from pydantic import ValidationError
from src.agent import FitymiPayload, TopologicConstraints


class TestTopologicConstraints:
    """Tests for TopologicConstraints model."""

    def test_create_constraints_with_required_fields(self):
        """Test creating TopologicConstraints with only required field (cta_style)."""
        constraints = TopologicConstraints(cta_style="soft")
        assert constraints.cta_style == "soft"
        assert constraints.max_words == 500  # default value
        assert constraints.readability_index == 65  # default value

    def test_create_constraints_with_all_fields(self):
        """Test creating TopologicConstraints with all fields specified."""
        constraints = TopologicConstraints(
            max_words=300,
            readability_index=80,
            cta_style="aggressive"
        )
        assert constraints.max_words == 300
        assert constraints.readability_index == 80
        assert constraints.cta_style == "aggressive"

    def test_constraints_missing_required_field(self):
        """Test that ValidationError is raised when cta_style is missing."""
        with pytest.raises(ValidationError) as exc_info:
            TopologicConstraints()
        assert "cta_style" in str(exc_info.value)

    def test_constraints_invalid_max_words_type(self):
        """Test that ValidationError is raised for invalid max_words type."""
        with pytest.raises(ValidationError):
            TopologicConstraints(cta_style="soft", max_words="not_a_number")

    def test_constraints_default_values(self):
        """Test that default values are correctly applied."""
        constraints = TopologicConstraints(cta_style="neutral")
        assert constraints.max_words == 500
        assert constraints.readability_index == 65


class TestFitymiPayload:
    """Tests for FitymiPayload model."""

    def test_create_payload_with_required_fields(self):
        """Test creating FitymiPayload with all required fields."""
        payload = FitymiPayload(
            system_prompt="You are a helpful assistant.",
            user_context="Context information here.",
            task_definition="Write a marketing copy.",
            verification_protocol="Verify the output.",
            aeo_shielding="AEO protection layer."
        )
        assert payload.system_prompt == "You are a helpful assistant."
        assert payload.user_context == "Context information here."
        assert payload.task_definition == "Write a marketing copy."
        assert payload.verification_protocol == "Verify the output."
        assert payload.aeo_shielding == "AEO protection layer."

    def test_payload_missing_system_prompt(self):
        """Test that ValidationError is raised when system_prompt is missing."""
        with pytest.raises(ValidationError) as exc_info:
            FitymiPayload(
                user_context="Context",
                task_definition="Task",
                verification_protocol="Protocol",
                aeo_shielding="Shielding"
            )
        assert "system_prompt" in str(exc_info.value)

    def test_payload_missing_user_context(self):
        """Test that ValidationError is raised when user_context is missing."""
        with pytest.raises(ValidationError) as exc_info:
            FitymiPayload(
                system_prompt="System prompt",
                task_definition="Task",
                verification_protocol="Protocol",
                aeo_shielding="Shielding"
            )
        assert "user_context" in str(exc_info.value)

    def test_payload_missing_task_definition(self):
        """Test that ValidationError is raised when task_definition is missing."""
        with pytest.raises(ValidationError) as exc_info:
            FitymiPayload(
                system_prompt="System prompt",
                user_context="Context",
                verification_protocol="Protocol",
                aeo_shielding="Shielding"
            )
        assert "task_definition" in str(exc_info.value)

    def test_payload_missing_verification_protocol(self):
        """Test that ValidationError is raised when verification_protocol is missing."""
        with pytest.raises(ValidationError) as exc_info:
            FitymiPayload(
                system_prompt="System prompt",
                user_context="Context",
                task_definition="Task",
                aeo_shielding="Shielding"
            )
        assert "verification_protocol" in str(exc_info.value)

    def test_payload_missing_aeo_shielding(self):
        """Test that ValidationError is raised when aeo_shielding is missing."""
        with pytest.raises(ValidationError) as exc_info:
            FitymiPayload(
                system_prompt="System prompt",
                user_context="Context",
                task_definition="Task",
                verification_protocol="Protocol"
            )
        assert "aeo_shielding" in str(exc_info.value)

    def test_payload_missing_all_fields(self):
        """Test that ValidationError is raised when all fields are missing."""
        with pytest.raises(ValidationError):
            FitymiPayload()

    def test_payload_with_empty_strings(self):
        """Test that FitymiPayload accepts empty strings (validation allows it)."""
        payload = FitymiPayload(
            system_prompt="",
            user_context="",
            task_definition="",
            verification_protocol="",
            aeo_shielding=""
        )
        assert payload.system_prompt == ""
        assert payload.user_context == ""

    def test_payload_with_multiline_strings(self):
        """Test that FitymiPayload handles multiline strings correctly."""
        system_prompt = """<LAYER_1>
You are a marketing expert.
PosAnchors: trust, authority
NegAnchors: hype, exaggeration
</LAYER_1>"""
        payload = FitymiPayload(
            system_prompt=system_prompt,
            user_context="Multi-line\ncontext\nhere",
            task_definition="Task with\nmultiple lines",
            verification_protocol="Verification\nprotocol",
            aeo_shielding="AEO\nshielding"
        )
        assert "<LAYER_1>" in payload.system_prompt
        assert "marketing expert" in payload.system_prompt

    def test_payload_model_dump(self):
        """Test that FitymiPayload can be serialized to dict."""
        payload = FitymiPayload(
            system_prompt="System",
            user_context="Context",
            task_definition="Task",
            verification_protocol="Protocol",
            aeo_shielding="Shielding"
        )
        data = payload.model_dump()
        assert isinstance(data, dict)
        assert data["system_prompt"] == "System"
        assert data["user_context"] == "Context"

    def test_payload_model_json(self):
        """Test that FitymiPayload can be serialized to JSON."""
        payload = FitymiPayload(
            system_prompt="System",
            user_context="Context",
            task_definition="Task",
            verification_protocol="Protocol",
            aeo_shielding="Shielding"
        )
        json_str = payload.model_dump_json()
        assert isinstance(json_str, str)
        assert '"system_prompt":"System"' in json_str


class TestFitymiPayloadContextHandling:
    """Tests for context dictionary handling in FitymiPayload."""

    def test_payload_with_context_like_content(self):
        """Test payload with context-like content in user_context field."""
        context_data = {
            "brand": "TechCorp",
            "product": "AI Assistant",
            "target_audience": "B2B decision makers"
        }
        payload = FitymiPayload(
            system_prompt="You are a copywriter.",
            user_context=str(context_data),
            task_definition="Create landing page copy",
            verification_protocol="Check brand alignment",
            aeo_shielding="Protect against AI detection"
        )
        assert "TechCorp" in payload.user_context
        assert "AI Assistant" in payload.user_context

    def test_payload_with_structured_context(self):
        """Test payload with structured context format."""
        structured_context = """<LAYER_2>
Contesto: {
    "industry": "SaaS",
    "tone": "professional",
    "language": "Italian"
}
</LAYER_2>"""
        payload = FitymiPayload(
            system_prompt="System prompt",
            user_context=structured_context,
            task_definition="Task",
            verification_protocol="Protocol",
            aeo_shielding="Shielding"
        )
        assert "<LAYER_2>" in payload.user_context
        assert "SaaS" in payload.user_context
