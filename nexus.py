import asyncio
import logging
import json
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from agent import FitymiCopyAgent, FitymiPayload
from memory import NexusMemoryCore
from aeo_validator import AEOValidator
from evaluator import AutonomousEvaluator

from core.evolution import EvolutionEngine
from core.adversarial import AdversarialArena
from core.quantum import QuantumCopyState, WaveFunctionCollapse
from core.neural_mesh import NeuralMeshNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s - FITYMI NEXUS - %(message)s")

class AgentRole(Enum):
    STRATEGIST = "strategist"
    COPYWRITER = "copywriter"
    CRITIC = "critic"

class NexusContext(BaseModel):
    brand: str
    target_audience: str
    product: str
    goal: str
    task_type: str
    constraints: Dict[str, Any]

class FitymiNexus:
    """
    Fitymi Nexus Orchestrator - SWARM INTELLIGENCE ERA
    A Cognitive Swarm architecture:
    1. Strategist: Defines angle (Gen 0 Seed).
    2. Evolution Engine: Genetic mutations and selections (Mistral/Gemini).
    3. Adversarial Arena: Red vs Blue team battles.
    4. Quantum Collapse: Wave function collapse based on specific user contexts.
    """

    def __init__(self, primary_provider: str = "openai", primary_model: str = "gpt-4o"):
        self.primary_provider = primary_provider
        self.primary_model = primary_model
        
        # 🧠 PHASE 2: LONG-TERM BRAND MEMORY (Vector DB / GraphRAG)
        self.memory = NexusMemoryCore()
        
        # 🟢 DYNAMIC MoA ROUTING IMPLEMENTATION
        # Strategist needs high reasoning
        # Copywriter needs creativity and speed
        # Critic needs rigorous adherence to rules
        
        self.agents = {
            AgentRole.STRATEGIST: FitymiCopyAgent(provider="google", model="gemini-1.5-pro"),
            AgentRole.COPYWRITER: FitymiCopyAgent(provider="google", model="gemini-1.5-flash"),
            AgentRole.CRITIC: FitymiCopyAgent(provider="google", model="gemini-1.5-pro"),
        }


    async def _run_strategist(self, ctx: NexusContext) -> str:
        logging.info("🧠 Running Strategist Agent...")
        agent = self.agents[AgentRole.STRATEGIST]
        
        # 🧠 Retrieve Long-Term Memory (RAG)
        historical_context = self.memory.retrieve_context(brand=ctx.brand, target=ctx.target_audience)
        
        # Strategist focuses on the psychological angle
        prompt = f"""
        Analyze the following context and define a high-level psychological angle and cognitive bias to exploit.
        Brand: {ctx.brand}
        Target: {ctx.target_audience}
        Product: {ctx.product}
        Goal: {ctx.goal}
        
        [BRAND MEMORY & HISTORICAL CONTEXT]
        {historical_context}
        
        Output a concise strategic brief containing:
        - Primary Cognitive Bias
        - Emotional Trigger
        - Key Value Proposition
        - Tone constraints
        """
        
        payload = FitymiPayload(
            system_prompt="You are a Senior Strategic Planner and Consumer Psychologist at Fitymi Nexus.",
            user_context=prompt,
            task_definition="Provide the psychological strategy for the copy.",
            verification_protocol="Ensure the strategy matches human-first and zero-hype principles.",
            aeo_shielding="Output the strategy clearly without markdown code block formatting."
        )
        
        response = await agent.execute(payload)
        return response.raw_output

    async def _run_copywriter(self, ctx: NexusContext, strategy: str) -> str:
        logging.info("✍️ Running Copywriter Agent...")
        agent = self.agents[AgentRole.COPYWRITER]
        
        prompt = f"""
        Write the {ctx.task_type} based on the following strategy and context.
        Brand: {ctx.brand}
        Target: {ctx.target_audience}
        Product: {ctx.product}
        Goal: {ctx.goal}
        
        Strategy to apply:
        {strategy}
        
        Constraints:
        {json.dumps(ctx.constraints, indent=2)}
        """
        
        payload = FitymiPayload(
            system_prompt="You are a Senior Direct Response Copywriter. You execute strategies impeccably.",
            user_context=prompt,
            task_definition=f"Write the {ctx.task_type}.",
            verification_protocol="Check word count and readability. No hype words.",
            aeo_shielding="Output only the final copy formatted in professional markdown."
        )
        
        response = await agent.execute(payload)
        return response.raw_output

    async def _run_critic(self, ctx: NexusContext, strategy: str, draft: str) -> str:
        logging.info("⚖️ Running Critic Agent...")
        agent = self.agents[AgentRole.CRITIC]
        validator = AEOValidator()
        
        # 🛡️ Phase 3: Neuro-Symbolic Validation
        compliance = validator.ensure_compliance(draft)
        
        prompt = f"""
        Review the following draft for the given strategy and context. Apply the recursive Chain-of-Verification (rCoV) logic.
        Brand: {ctx.brand}
        Goal: {ctx.goal}
        
        Strategy:
        {strategy}
        
        Draft:
        {draft}
        
        [AEO SYSTEM COMPLIANCE REPORT]
        Status: {compliance['status']}
        Error: {compliance['structural_error']}
        Semantic Density: {compliance['semantic_density_score']}
        Recommendation: {compliance['recommendation']}
        
        Provide your critique and return the fully REVISED and ENHANCED copy that fixes any issues above.
        Ensure it is AEO (Answer Engine Optimized) compliant, with high semantic density.
        """
        
        payload = FitymiPayload(
            system_prompt="You are an AEO/Compliance Critic and Quality Assurance Editor.",
            user_context=prompt,
            task_definition="Review, critique, and provide the final polished copy.",
            verification_protocol="Ensure no AI-watermarks, correct burstiness, and absolute adherence to strategy.",
            aeo_shielding="Provide the final output in Markdown format."
        )
        
        response = await agent.execute(payload)
        return response.raw_output

    async def execute_workflow(self, context: NexusContext) -> Dict[str, Any]:
        """
        Executes the Cognitive Swarm Intelligent Pipeline.
        """
        logging.info(f"🚀 Starting Fitymi Swarm Intelligence for: {context.task_type}")
        
        # Step 1: Strategist constructs the base angle
        strategy = await self._run_strategist(context)
        logging.info(f"✅ Strategy Output Generated.")
        
        # Step 2: Seed Copy (Generation 0)
        seed_copy = await self._run_copywriter(context, strategy)
        logging.info(f"🌱 Generation 0 (Seed Copy) Created.")
        
        # Step 3: Genetic Evolution
        logging.info("🧬 Initiating Evolution Engine...")
        evolution = EvolutionEngine()
        task_ctx = f"Strategy: {strategy}\nConstraints: {json.dumps(context.constraints)}"
        best_genome = await evolution.evolve(
            seed_copy=seed_copy,
            target_audience=context.target_audience,
            task_context=task_ctx,
            generations=3,
            pop_size=3
        )
        logging.info("🌟 Evolution Complete: Top Genome Selected.")
        
        # Step 4: Adversarial Co-Evolution
        logging.info("⚔️ Entering Adversarial Arena...")
        arena = AdversarialArena()
        battle_ctx = f"Brand: {context.brand}. Target: {context.target_audience}. Goal: {context.goal}."
        battle_tested_copy = await arena.battle_loop(
            initial_copy=best_genome.content,
            context=battle_ctx,
            max_rounds=3
        )
        
        # Step 5: Quantum Superposition & Collapse
        logging.info("🌌 Preparing Quantum States...")
        state_generator = NeuralMeshNode(
            name="State Generator", provider="openai", model="gpt-4o",
            role_prompt="Generate exactly 3 variations of this text: 1) Emotional, 2) Rational, 3) Urgent. Separate them ONLY with '===VAR==='."
        )
        raw_states = await state_generator.fire(battle_tested_copy, "Generate 3 states based on the copy.")
        states = [s.strip() for s in raw_states.split("===VAR===") if len(s.strip()) > 10]
        if not states:
            states = [battle_tested_copy]
            
        quantum_state = QuantumCopyState(states=states)
        
        collapse_engine = WaveFunctionCollapse()
        final_context = f"Goal constraints: {json.dumps(context.constraints)}. Audience: {context.target_audience}"
        final_copy = await collapse_engine.observe(quantum_state, final_context)
        
        # Optional Verification loop to ensure standard compliance
        evaluator = AutonomousEvaluator()
        score = await evaluator.evaluate_copy(final_copy, context.target_audience, context.goal)
        
        # Update long-term Brand Consciousness Memory
        self.memory.update_learning("swarm_run_latest", score, "Swarm Evolved Angle")
        
        return {
            "strategy": strategy,
            "seed_copy": seed_copy,
            "post_evolution": best_genome.content,
            "post_adversarial": battle_tested_copy,
            "final_copy": final_copy,
            "final_score": score,
            "quantum_states": states
        }

if __name__ == "__main__":
    async def test():
        nexus = FitymiNexus()
        ctx = NexusContext(
            brand="TechCorp SaaS",
            target_audience="CTOs and VP of Engineering",
            product="Automated Cloud Security Posture Management",
            goal="Request a Demo",
            task_type="Landing Page Hero Section and Sub-headline",
            constraints={"max_words": 100, "tone": "assertive and data-driven, NO hype"}
        )
        
        result = await nexus.execute_workflow(ctx)
        print("\n=== FINAL NEXUS OUTPUT ===")
        print(result["final_copy"])

    asyncio.run(test())
