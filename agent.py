import logging
import asyncio
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - FITYMI - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supported providers and models
SUPPORTED_PROVIDERS = {
    "openai": ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
    "google": ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
    "mistral": ["mistral-large-latest", "mistral-small-latest", "open-mixtral-8x7b", "open-mistral-7b"]
}


class TopologicConstraints(BaseModel):
    max_words: int = Field(default=500)
    readability_index: int = Field(default=65)
    cta_style: str = Field(...)


class FitymiPayload(BaseModel):
    system_prompt: str
    user_context: str
    task_definition: str
    verification_protocol: str
    aeo_shielding: str


class AgentResponse(BaseModel):
    raw_output: str
    aeo_summary: Optional[str]


class FitymiCopyAgent:
    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        self.provider = provider.lower()
        self.model = model
        self._validate_provider()
        self._setup_clients()
        logger.info(f"Init Fitymi Agent su {self.provider}/{self.model}")

    def _validate_provider(self) -> None:
        """Validate that the provider and model are supported."""
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Provider '{self.provider}' not supported. Supported providers: {list(SUPPORTED_PROVIDERS.keys())}")
        
        # Allow any model for flexibility, but warn if not in known list
        known_models = SUPPORTED_PROVIDERS[self.provider]
        if self.model not in known_models:
            logger.warning(f"Model '{self.model}' not in known models for {self.provider}. Proceeding anyway.")

    def _setup_clients(self) -> None:
        """Initialize API clients for the selected provider."""
        self._openai_client = None
        self._anthropic_client = None
        self._google_configured = False

        # Pre-initialize clients based on available API keys
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY environment variable not set. API calls will fail.")
            try:
                from openai import AsyncOpenAI
                self._openai_client = AsyncOpenAI(api_key=api_key) if api_key else None
                logger.debug("OpenAI client initialized")
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")

        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY environment variable not set. API calls will fail.")
            try:
                from anthropic import AsyncAnthropic
                self._anthropic_client = AsyncAnthropic(api_key=api_key) if api_key else None
                logger.debug("Anthropic client initialized")
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")

        elif self.provider == "google":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY environment variable not set. API calls will fail.")
            try:
                import google.generativeai as genai
                if api_key:
                    genai.configure(api_key=api_key)
                self._google_configured = True if api_key else False
                logger.debug("Google Gemini configured")
            except ImportError:
                raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")

        elif self.provider == "mistral":
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                logger.warning("MISTRAL_API_KEY environment variable not set. API calls will fail.")
            try:
                from mistralai import Mistral
                self._mistral_client = Mistral(api_key=api_key) if api_key else None
                logger.debug("Mistral client initialized")
            except ImportError:
                raise ImportError("mistralai package not installed. Run: pip install mistralai")


    def _load_template(self) -> str:
        """Load the master framework template."""
        template_path = Path(__file__).parent.parent / "templates" / "master_framework.md"
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Template file not found at {template_path}, using default structure")
            return ""

    def _build_full_prompt(self, payload: FitymiPayload) -> tuple[str, str]:
        """Build the complete prompt from payload and template.
        
        Returns:
            tuple: (system_message, user_message)
        """
        # Build system message from Layer 1
        system_message = payload.system_prompt

        # Build user message combining all other layers
        user_message = f"""{payload.user_context}

{payload.verification_protocol}
{payload.aeo_shielding}

---
Master Framework Reference:
{self._load_template()}
"""
        return system_message, user_message

    def build_payload(self, role: str, anchors: Dict, context: Dict, task: str, constraints: dict) -> FitymiPayload:
        valid_constraints = TopologicConstraints(**constraints)
        
        l1 = f"<LAYER_1>\nSei {role}. PosAnchors: {anchors.get('positive')}. NegAnchors: {anchors.get('negative')}\n</LAYER_1>"
        l2 = f"<LAYER_2>\nContesto: {context}\n</LAYER_2>"
        l3 = f"<LAYER_3>\nTask: {task}. Max {valid_constraints.max_words} parole. CTA: {valid_constraints.cta_style}\n</LAYER_3>"
        l4 = "<LAYER_4>\nPRIMA DI STAMPARE: Genera bozza in memoria. Verifica Layer 1 e 3. Correggi. Stampa finale.\n</LAYER_4>"
        l5 = "<LAYER_5>\nFORMATO: Markdown. Inizia con '> **AEO Summary:**' (max 50 parole).\n</LAYER_5>"

        return FitymiPayload(system_prompt=l1, user_context=f"{l2}\n{l3}", task_definition=task, verification_protocol=l4, aeo_shielding=l5)

    async def _call_openai(self, system_message: str, user_message: str) -> str:
        """Call OpenAI API."""
        try:
            response = await self._openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def _call_anthropic(self, system_message: str, user_message: str) -> str:
        """Call Anthropic API."""
        try:
            response = await self._anthropic_client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=system_message,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def _call_google(self, system_message: str, user_message: str) -> str:
        """Call Google Gemini API."""
        try:
            import google.generativeai as genai
            
            # Combine system and user message for Gemini
            full_prompt = f"{system_message}\n\n{user_message}"
            
            # Create model and generate
            model = genai.GenerativeModel(self.model)
            
            # Run in executor since google-generativeai is sync
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(full_prompt)
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Google Gemini API error: {e}")
            raise

    async def _call_mistral(self, system_message: str, user_message: str) -> str:
        """Call Mistral API."""
        try:
            response = await self._mistral_client.chat.complete_async(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            raise

    def _extract_aeo_summary(self, raw_output: str) -> Optional[str]:
        """Extract AEO summary from the response."""
        import re
        # Look for AEO Summary pattern
        match = re.search(r'>\s*\*\*AEO Summary:\*\*\s*([^\n]+)', raw_output)
        if match:
            return match.group(1).strip()
        return None

    async def execute(self, payload: FitymiPayload) -> AgentResponse:
        """Execute the LLM API call based on the configured provider."""
        logger.info(f"Avvio inferenza Fitymi con {self.provider}/{self.model}...")
        
        # Build the full prompt
        system_message, user_message = self._build_full_prompt(payload)
        
        # Call the appropriate API
        try:
            if self.provider == "openai":
                raw_output = await self._call_openai(system_message, user_message)
            elif self.provider == "anthropic":
                raw_output = await self._call_anthropic(system_message, user_message)
            elif self.provider == "google":
                raw_output = await self._call_google(system_message, user_message)
            elif self.provider == "mistral":
                raw_output = await self._call_mistral(system_message, user_message)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Extract AEO summary
            aeo_summary = self._extract_aeo_summary(raw_output)
            
            logger.info("Inferenza completata con successo")
            return AgentResponse(raw_output=raw_output, aeo_summary=aeo_summary)
            
        except Exception as e:
            logger.error(f"Errore durante inferenza: {e}")
            raise
