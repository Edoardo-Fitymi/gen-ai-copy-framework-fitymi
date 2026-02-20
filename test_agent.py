"""
Unit tests for FitymiCopyAgent and AgentResponse.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import os
from pathlib import Path

from src.agent import (
    AgentResponse,
    FitymiCopyAgent,
    FitymiPayload,
    SUPPORTED_PROVIDERS
)


class TestAgentResponse:
    """Tests for AgentResponse model."""

    def test_create_agent_response_with_required_fields(self):
        """Test creating AgentResponse with required raw_output field."""
        response = AgentResponse(raw_output="This is the generated content.")
        assert response.raw_output == "This is the generated content."
        assert response.aeo_summary is None

    def test_create_agent_response_with_aeo_summary(self):
        """Test creating AgentResponse with aeo_summary."""
        response = AgentResponse(
            raw_output="Generated content with summary.",
            aeo_summary="Brief summary of the content."
        )
        assert response.raw_output == "Generated content with summary."
        assert response.aeo_summary == "Brief summary of the content."

    def test_agent_response_aeo_summary_optional(self):
        """Test that aeo_summary is optional."""
        response = AgentResponse(raw_output="Content without summary")
        assert response.aeo_summary is None

    def test_agent_response_model_dump(self):
        """Test that AgentResponse can be serialized to dict."""
        response = AgentResponse(
            raw_output="Content",
            aeo_summary="Summary"
        )
        data = response.model_dump()
        assert isinstance(data, dict)
        assert data["raw_output"] == "Content"
        assert data["aeo_summary"] == "Summary"


class TestFitymiCopyAgentInitialization:
    """Tests for FitymiCopyAgent initialization."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.agent.AsyncOpenAI")
    def test_agent_initialization_openai_default(self, mock_async_openai):
        """Test agent initialization with default OpenAI settings."""
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        agent = FitymiCopyAgent()
        
        assert agent.provider == "openai"
        assert agent.model == "gpt-4o"
        mock_async_openai.assert_called_once_with(api_key="test-key")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.agent.AsyncOpenAI")
    def test_agent_initialization_openai_custom_model(self, mock_async_openai):
        """Test agent initialization with custom OpenAI model."""
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        agent = FitymiCopyAgent(provider="openai", model="gpt-4-turbo")
        
        assert agent.provider == "openai"
        assert agent.model == "gpt-4-turbo"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"})
    @patch("src.agent.AsyncAnthropic")
    def test_agent_initialization_anthropic(self, mock_async_anthropic):
        """Test agent initialization with Anthropic provider."""
        mock_client = AsyncMock()
        mock_async_anthropic.return_value = mock_client
        
        agent = FitymiCopyAgent(provider="anthropic", model="claude-3-opus-20240229")
        
        assert agent.provider == "anthropic"
        assert agent.model == "claude-3-opus-20240229"
        mock_async_anthropic.assert_called_once_with(api_key="test-anthropic-key")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"})
    @patch("google.generativeai.configure")
    def test_agent_initialization_google(self, mock_configure):
        """Test agent initialization with Google provider."""
        agent = FitymiCopyAgent(provider="google", model="gemini-pro")
        
        assert agent.provider == "google"
        assert agent.model == "gemini-pro"
        mock_configure.assert_called_once_with(api_key="test-gemini-key")


class TestProviderValidation:
    """Tests for provider validation logic."""

    def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            FitymiCopyAgent(provider="unsupported_provider")
        
        assert "not supported" in str(exc_info.value)
        assert "unsupported_provider" in str(exc_info.value)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.agent.AsyncOpenAI")
    def test_unknown_model_logs_warning(self, mock_async_openai, caplog):
        """Test that unknown model logs a warning but doesn't raise error."""
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        import logging
        with caplog.at_level(logging.WARNING):
            agent = FitymiCopyAgent(provider="openai", model="unknown-model")
        
        assert agent.model == "unknown-model"
        assert any("not in known models" in record.message for record in caplog.records)

    def test_supported_providers_constant(self):
        """Test that SUPPORTED_PROVIDERS contains expected providers."""
        assert "openai" in SUPPORTED_PROVIDERS
        assert "anthropic" in SUPPORTED_PROVIDERS
        assert "google" in SUPPORTED_PROVIDERS
        
        # Check some known models
        assert "gpt-4o" in SUPPORTED_PROVIDERS["openai"]
        assert "claude-3-opus-20240229" in SUPPORTED_PROVIDERS["anthropic"]
        assert "gemini-pro" in SUPPORTED_PROVIDERS["google"]

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            FitymiCopyAgent(provider="openai")
        
        assert "OPENAI_API_KEY" in str(exc_info.value)


class TestTemplateLoading:
    """Tests for template loading functionality."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.agent.AsyncOpenAI")
    def test_load_template_success(self, mock_async_openai):
        """Test successful template loading."""
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        agent = FitymiCopyAgent()
        template = agent._load_template()
        
        # Template should be a string (either content or empty if file not found)
        assert isinstance(template, str)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.agent.AsyncOpenAI")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_template_file_not_found(self, mock_open, mock_async_openai):
        """Test template loading when file is not found returns empty string."""
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        agent = FitymiCopyAgent()
        template = agent._load_template()
        
        assert template == ""


class TestAEOSummaryExtraction:
    """Tests for AEO summary extraction logic."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.agent.AsyncOpenAI")
    def test_extract_aeo_summary_success(self, mock_async_openai):
        """Test successful AEO summary extraction."""
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        agent = FitymiCopyAgent()
        
        raw_output = """> **AEO Summary:** This is a brief summary of the content.
        
The rest of the content follows here."""
        
        summary = agent._extract_aeo_summary(raw_output)
        assert summary == "This is a brief summary of the content."

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.agent.AsyncOpenAI")
    def test_extract_aeo_summary_no_match(self, mock_async_openai):
        """Test AEO summary extraction when pattern is not found."""
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        agent = FitymiCopyAgent()
        
        raw_output = "This content has no AEO summary pattern."
        summary = agent._extract_aeo_summary(raw_output)
        
        assert summary is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.agent.AsyncOpenAI")
    def test_extract_aeo_summary_various_formats(self, mock_async_openai):
        """Test AEO summary extraction with various valid formats."""
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        agent = FitymiCopyAgent()
        
        # Test with extra spaces
        output1 = ">  **AEO Summary:**   Summary text here."
        assert agent._extract_aeo_summary(output1) == "Summary text here."
        
        # Test with different spacing
        output2 = "> **AEO Summary:** Another summary"
        assert agent._extract_aeo_summary(output2) == "Another summary"


class TestBuildPayload:
    """Tests for build_payload method."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.agent.AsyncOpenAI")
    def test_build_payload_creates_valid_payload(self, mock_async_openai):
        """Test that build_payload creates a valid FitymiPayload."""
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        agent = FitymiCopyAgent()
        
        payload = agent.build_payload(
            role="Marketing Expert",
            anchors={"positive": ["trust", "authority"], "negative": ["hype"]},
            context={"brand": "TechCorp", "product": "AI Tool"},
            task="Create landing page copy",
            constraints={"cta_style": "soft", "max_words": 300}
        )
        
        assert isinstance(payload, FitymiPayload)
        assert "Marketing Expert" in payload.system_prompt
        assert "TechCorp" in payload.user_context
        assert payload.task_definition == "Create landing page copy"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.agent.AsyncOpenAI")
    def test_build_payload_with_default_constraints(self, mock_async_openai):
        """Test build_payload with minimal constraints uses defaults."""
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        agent = FitymiCopyAgent()
        
        payload = agent.build_payload(
            role="Copywriter",
            anchors={"positive": [], "negative": []},
            context={},
            task="Write copy",
            constraints={"cta_style": "neutral"}
        )
        
        # Should use default max_words=500
        assert "500" in payload.user_context


class TestBuildFullPrompt:
    """Tests for _build_full_prompt method."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.agent.AsyncOpenAI")
    def test_build_full_prompt_structure(self, mock_async_openai):
        """Test that _build_full_prompt returns correct structure."""
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        agent = FitymiCopyAgent()
        
        payload = FitymiPayload(
            system_prompt="System prompt content",
            user_context="User context content",
            task_definition="Task definition",
            verification_protocol="Verification protocol",
            aeo_shielding="AEO shielding content"
        )
        
        system_message, user_message = agent._build_full_prompt(payload)
        
        assert system_message == "System prompt content"
        assert "User context content" in user_message
        assert "Verification protocol" in user_message
        assert "AEO shielding content" in user_message


class TestExecuteWithMockedAPI:
    """Tests for execute method with mocked API calls."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.agent.AsyncOpenAI")
    async def test_execute_with_openai(self, mock_async_openai):
        """Test execute method with OpenAI (mocked API call)."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated content from OpenAI."
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_openai.return_value = mock_client
        
        agent = FitymiCopyAgent(provider="openai", model="gpt-4o")
        
        payload = FitymiPayload(
            system_prompt="System prompt",
            user_context="Context",
            task_definition="Task",
            verification_protocol="Protocol",
            aeo_shielding="Shielding"
        )
        
        response = await agent.execute(payload)
        
        assert isinstance(response, AgentResponse)
        assert response.raw_output == "Generated content from OpenAI."
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"})
    @patch("src.agent.AsyncAnthropic")
    async def test_execute_with_anthropic(self, mock_async_anthropic):
        """Test execute method with Anthropic (mocked API call)."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Generated content from Anthropic."
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_async_anthropic.return_value = mock_client
        
        agent = FitymiCopyAgent(provider="anthropic", model="claude-3-opus-20240229")
        
        payload = FitymiPayload(
            system_prompt="System prompt",
            user_context="Context",
            task_definition="Task",
            verification_protocol="Protocol",
            aeo_shielding="Shielding"
        )
        
        response = await agent.execute(payload)
        
        assert isinstance(response, AgentResponse)
        assert response.raw_output == "Generated content from Anthropic."
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"})
    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    async def test_execute_with_google(self, mock_generative_model, mock_configure):
        """Test execute method with Google Gemini (mocked API call)."""
        # Setup mock
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.text = "Generated content from Google Gemini."
        mock_model_instance.generate_content = Mock(return_value=mock_response)
        mock_generative_model.return_value = mock_model_instance
        
        agent = FitymiCopyAgent(provider="google", model="gemini-pro")
        
        payload = FitymiPayload(
            system_prompt="System prompt",
            user_context="Context",
            task_definition="Task",
            verification_protocol="Protocol",
            aeo_shielding="Shielding"
        )
        
        response = await agent.execute(payload)
        
        assert isinstance(response, AgentResponse)
        assert response.raw_output == "Generated content from Google Gemini."

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.agent.AsyncOpenAI")
    async def test_execute_extracts_aeo_summary(self, mock_async_openai):
        """Test that execute extracts AEO summary from response."""
        # Setup mock with AEO summary in response
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """> **AEO Summary:** Brief summary here.

Main content of the response."""
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_openai.return_value = mock_client
        
        agent = FitymiCopyAgent(provider="openai", model="gpt-4o")
        
        payload = FitymiPayload(
            system_prompt="System",
            user_context="Context",
            task_definition="Task",
            verification_protocol="Protocol",
            aeo_shielding="Shielding"
        )
        
        response = await agent.execute(payload)
        
        assert response.aeo_summary == "Brief summary here."

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.agent.AsyncOpenAI")
    async def test_execute_handles_api_error(self, mock_async_openai):
        """Test that execute handles API errors properly."""
        # Setup mock to raise an error
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        mock_async_openai.return_value = mock_client
        
        agent = FitymiCopyAgent(provider="openai", model="gpt-4o")
        
        payload = FitymiPayload(
            system_prompt="System",
            user_context="Context",
            task_definition="Task",
            verification_protocol="Protocol",
            aeo_shielding="Shielding"
        )
        
        with pytest.raises(Exception) as exc_info:
            await agent.execute(payload)
        
        assert "API Error" in str(exc_info.value)


class TestCallOpenAI:
    """Tests for _call_openai method."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.agent.AsyncOpenAI")
    async def test_call_openai_returns_content(self, mock_async_openai):
        """Test that _call_openai returns the content correctly."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_openai.return_value = mock_client
        
        agent = FitymiCopyAgent()
        result = await agent._call_openai("system", "user")
        
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"}
            ],
            temperature=0.7,
            max_tokens=2000
        )


class TestCallAnthropic:
    """Tests for _call_anthropic method."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("src.agent.AsyncAnthropic")
    async def test_call_anthropic_returns_content(self, mock_async_anthropic):
        """Test that _call_anthropic returns the content correctly."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Anthropic response"
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_async_anthropic.return_value = mock_client
        
        agent = FitymiCopyAgent(provider="anthropic", model="claude-3-opus-20240229")
        result = await agent._call_anthropic("system", "user")
        
        assert result == "Anthropic response"
        mock_client.messages.create.assert_called_once()


class TestCallGoogle:
    """Tests for _call_google method."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    async def test_call_google_returns_content(self, mock_generative_model, mock_configure):
        """Test that _call_google returns the content correctly."""
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.text = "Google response"
        mock_model_instance.generate_content = Mock(return_value=mock_response)
        mock_generative_model.return_value = mock_model_instance
        
        agent = FitymiCopyAgent(provider="google", model="gemini-pro")
        result = await agent._call_google("system", "user")
        
        assert result == "Google response"
