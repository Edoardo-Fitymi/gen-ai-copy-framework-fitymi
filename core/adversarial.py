import asyncio
import logging
from typing import List

from core.neural_mesh import NeuralMeshNode

logger = logging.getLogger(__name__)

class AdversarialArena:
    """Implement Adversarial Co-Evolution (Red Team vs Blue Team)."""
    
    def __init__(self):
        # Red Team: Uses fast but aggressive logic to find flaws
        self.red_team = NeuralMeshNode(
            name="Red Team Critic",
            provider="mistral",
            model="open-mistral-7b",
            role_prompt=(
                "You are the Red Team Marketing Critic. Your job is to aggressively attack "
                "the provided copy. Find logical flaws, hype-words, boring tropes, or lack "
                "of clarity. You do not fix it, you only attack it. Keep it brief and list "
                "top 3 lethal flaws."
            )
        )
        
        # Blue Team: Uses Deep Reasoning to fix flaws and improve
        self.blue_team = NeuralMeshNode(
            name="Blue Team Defender",
            provider="google",
            model="gemini-1.5-pro",
            role_prompt=(
                "You are the Blue Team Defender. You receive marketing copy and harsh criticisms. "
                "You must absorb the critiques and output a NEW, strictly superior version of the "
                "copy that resolves all attacks smoothly without sounding defensive. Output ONLY the new copy."
            )
        )

    async def battle_loop(self, initial_copy: str, context: str, max_rounds: int = 3) -> str:
        """Runs the zero-sum game between Red and Blue teams."""
        logger.info(f"⚔️ Starting Adversarial Battle Loop (Max {max_rounds} rounds)...")
        current_copy = initial_copy
        
        for round_num in range(1, max_rounds + 1):
            logger.info(f"🥊 Round {round_num} / {max_rounds}")
            
            # Red Team Attacks
            attack_prompt = f"Context: {context}\n\nCopy to attack:\n{current_copy}\n\nList vulnerabilities."
            critiques = await self.red_team.fire(attack_prompt, "Attack the copy.")
            
            # Early stopping heuristic 
            # If the critic cannot find 3 flaws easily or praises it, break the loop.
            critique_lower = critiques.lower()
            if "no major flaws" in critique_lower or "flawless" in critique_lower or len(critiques.split('\n')) < 2:
                logger.info("🛡️ Blue Team reached invincibility (Early Stopping). No lethal flaws found.")
                break
                
            logger.info(f"🔴 Red Team Critique:\n{critiques[:200]}...")
            
            # Blue Team Defends
            defend_prompt = f"Context: {context}\n\nCurrent Copy:\n{current_copy}\n\nCritiques to resolve:\n{critiques}\n\nProvide the revised copy."
            revised_copy = await self.blue_team.fire(defend_prompt, "Enhance the copy to survive attacks.")
            
            # Basic validation to ensure Blue team didn't output conversational filler
            if len(revised_copy) < 20: 
                logger.warning("Blue team output was too short, keeping previous copy.")
                break
                
            current_copy = revised_copy
            logger.info(f"🔵 Blue Team Defense generated. Copy length updated.")
            
        logger.info("🏁 Battle Loop concluded.")
        return current_copy
