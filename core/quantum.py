import asyncio
import logging
from typing import List

from core.neural_mesh import NeuralMeshNode

logger = logging.getLogger(__name__)

class QuantumCopyState:
    """Holds superimposed versions of the copy until collapse."""
    def __init__(self, states: List[str]):
        self.states = states
        
class ObserverNode(NeuralMeshNode):
    def __init__(self):
        super().__init__(
            name="Gemini-Pro Observer",
            provider="google",
            model="gemini-1.5-pro",
            role_prompt=(
                "You are the Quantum Observer. You receive multiple variations of a text and a "
                "specific, late-binding context. Your job is to select the SINGLE best variation "
                "that perfectly matches the context. Output exactly that variation and nothing else. "
                "Do not explain your choice."
            )
        )

class WaveFunctionCollapse:
    """Collapses the quantum state into a final copy based on observer context."""
    def __init__(self):
        self.observer = ObserverNode()

    async def observe(self, quantum_state: QuantumCopyState, final_context: str) -> str:
        logger.info(f"🌌 Collapsing Wave Function from {len(quantum_state.states)} states...")
        
        # If there's only one state, no need to collapse.
        if len(quantum_state.states) == 1:
            logger.info("Only one state present, auto-collapsing.")
            return quantum_state.states[0]
            
        # Build prompt
        prompt_parts = [f"FINAL CONTEXT/CONSTRAINTS: {final_context}\n\n== SUPERIMPOSED STATES =="]
        for i, state in enumerate(quantum_state.states):
            # Removing markdown format wrappers if they exist in state
            clean_state = state.strip('`').replace('markdown\n', '')
            prompt_parts.append(f"\n--- [STATE {i+1}] ---\n{clean_state}\n")
            
        prompt_parts.append("\nCollapse the wave function. Output ONLY the text of the best state for the context.")
        
        collapsed_copy = await self.observer.fire("\n".join(prompt_parts), "Select the best state.")
        
        # Clean output
        collapsed_copy = collapsed_copy.strip('`').replace('markdown\n', '').strip()
        logger.info(f"✨ Wave function collapsed successfully.")
        return collapsed_copy
