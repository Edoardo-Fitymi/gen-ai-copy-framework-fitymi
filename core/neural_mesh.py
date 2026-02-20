import asyncio
import time
import logging
from typing import List, Dict, Any, Optional

from agent import FitymiCopyAgent, FitymiPayload

logger = logging.getLogger(__name__)

class RateLimiter:
    """Async Rate Limiter using Token Bucket algorithm."""
    def __init__(self, rate: int, per: float):
        self.rate = rate
        self.per = per
        self.capacity = rate
        self.tokens = rate
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens + elapsed * (self.rate / self.per))
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            await asyncio.sleep(0.1)

# Global limiters
# Gemini API: 60 RPM free tier
GEMINI_LIMITER = RateLimiter(60, 60.0)
# Mistral API: Can be stricter on free tier, set to 1 req/sec safe limit
MISTRAL_LIMITER = RateLimiter(60, 60.0)
# OpenAI/Anthropic fallback limiters
OPENAI_LIMITER = RateLimiter(100, 60.0)

class NeuralMeshNode:
    """
    A single node in the swarm intelligence graph.
    Capable of receiving a signal, processing it via its LLM agent,
    and optionally propagating it to connected nodes.
    """
    def __init__(self, name: str, provider: str, model: str, role_prompt: str):
        self.name = name
        self.agent = FitymiCopyAgent(provider=provider, model=model)
        self.role_prompt = role_prompt
        self.connections: List['NeuralMeshNode'] = []
        self.activation_threshold = 0.7
        self.provider = provider
        
    async def _wait_for_rate_limit(self):
        if self.provider == "google":
            await GEMINI_LIMITER.acquire()
        elif self.provider == "mistral":
            await MISTRAL_LIMITER.acquire()
        elif self.provider == "openai":
            await OPENAI_LIMITER.acquire()

    def connect(self, node: 'NeuralMeshNode'):
        """Connect this node to downstream nodes."""
        self.connections.append(node)

    async def process(self, input_signal: str, task: str, constraints: Optional[Dict[str, Any]] = None) -> str:
        """Internal processing function for this node."""
        logger.info(f"🕸️ [Mesh Node: {self.name}] Processing signal...")
        await self._wait_for_rate_limit()
        
        # Customize payload depending on what the node does
        payload = FitymiPayload(
            system_prompt=self.role_prompt,
            user_context=input_signal,
            task_definition=task,
            verification_protocol="Ensure high quality and deep reasoning. Follow the constraints.",
            aeo_shielding="Output the result in plain format or markdown without conversational filler."
        )
        
        response = await self.agent.execute(payload)
        return response.raw_output

    async def fire(self, input_signal: str, task: str, constraints: Optional[Dict[str, Any]] = None) -> str:
        """
        Receives an activation signal.
        If logic warrants, propagates output to connected edges.
        """
        output = await self.process(input_signal, task, constraints)
        return output
