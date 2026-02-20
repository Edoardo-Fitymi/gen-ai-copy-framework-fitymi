import asyncio
import logging
import json
import random
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from core.neural_mesh import NeuralMeshNode

logger = logging.getLogger(__name__)

class CopyGenome(BaseModel):
    """The DNA of a piece of copy."""
    id: str
    content: str
    structural_genes: List[str] = Field(default_factory=list)
    emotional_genes: List[str] = Field(default_factory=list)
    conversion_genes: List[str] = Field(default_factory=list)
    fitness_score: float = 0.0

class EvolutionEngine:
    """Handles the Darwinian evolution of copy."""
    def __init__(self):
        # The Fast Scout Mutator
        self.mutator_node = NeuralMeshNode(
            name="Mistral-7B Mutator",
            provider="mistral",
            model="open-mistral-7b", 
            role_prompt="You are an Evolutionary Mutation Engine. You take a seed copy and produce strictly format-adherent variations."
        )
        # The Selection Filter
        self.selector_node = NeuralMeshNode(
            name="Gemini-Flash Selector",
            provider="google",
            model="gemini-1.5-flash",
            role_prompt="You are an AI Fitness Evaluator. You score variations based on impact, clarity, and conversion potential. Respond ONLY with valid JSON."
        )

    async def mutate(self, seed_content: str, num_variants: int = 3, task_context: str = "") -> List[CopyGenome]:
        logger.info(f"🧬 Mutating seed into {num_variants} variations...")
        task = f"Generate {num_variants} distinct, highly creative variations of this copy. Output them separated by '===VAR==='.\nTask constraints: {task_context}"
        
        raw_variants = await self.mutator_node.fire(seed_content, task)
        # Parse output
        variants_texts = [v.strip() for v in raw_variants.split("===VAR===") if len(v.strip()) > 10]
        
        genomes = []
        for i, text in enumerate(variants_texts[:num_variants]):
            genomes.append(CopyGenome(id=f"gen_v{i}_{random.randint(100,999)}", content=text))
            
        # Fallback if Mistral failed to split correctly
        if not genomes:
            genomes.append(CopyGenome(id="gen_v0_fallback", content=raw_variants))
            
        return genomes

    async def evaluate_fitness(self, genomes: List[CopyGenome], target_audience: str) -> List[CopyGenome]:
        logger.info(f"⚖️ Evaluating fitness of {len(genomes)} genomes...")
        
        async def evaluate_single(genome: CopyGenome):
            task = f"""Evaluate this copy for audience: '{target_audience}'. 
Return ONLY a JSON object: {{"emotional_impact": float, "clarity": float, "brand_alignment": float, "overall_score": float}} where values are 0.0 to 1.0."""
            result = await self.selector_node.fire(genome.content, task)
            try:
                # Clean up json format if wrapped in markdown
                clean_json = result.strip()
                if clean_json.startswith("```"):
                    lines = clean_json.split("\n")
                    if lines[0].startswith("```"): lines = lines[1:]
                    if lines[-1].startswith("```"): lines = lines[:-1]
                    clean_json = "\n".join(lines).strip()
                scores = json.loads(clean_json)
                genome.fitness_score = scores.get("overall_score", 0.5)
            except Exception as e:
                logger.warning(f"Failed to parse fitness JSON: {e}")
                genome.fitness_score = 0.1 # Penalty
            return genome

        evaluated = await asyncio.gather(*[evaluate_single(g) for g in genomes])
        return sorted(list(evaluated), key=lambda x: x.fitness_score, reverse=True)

    def crossover(self, parent1: CopyGenome, parent2: CopyGenome) -> CopyGenome:
        logger.info(f"🔀 Performing crossover between {parent1.id} and {parent2.id}...")
        
        # Simple heuristic crossover: Merge halves or sentences.
        words1 = parent1.content.split()
        words2 = parent2.content.split()
        
        split1 = len(words1) // 2
        split2 = len(words2) // 2
        
        child_content = " ".join(words1[:split1] + words2[split2:])
        child = CopyGenome(id=f"child_{random.randint(1000,9999)}", content=child_content)
        
        return child

    async def evolve(self, seed_copy: str, target_audience: str, task_context: str = "", generations: int = 3, pop_size: int = 3) -> CopyGenome:
        logger.info(f"🔄 Starting evolution loop for {generations} generations...")
        
        # Generation 0
        current_pop = [CopyGenome(id="seed", content=seed_copy)]
        
        for gen in range(1, generations + 1):
            logger.info(f"--- Generation {gen} ---")
            
            # Mutate top performer
            new_variants = await self.mutate(current_pop[0].content, num_variants=pop_size, task_context=task_context)
            
            # Add crossover child if we have multiple parents from previous gen
            if len(current_pop) >= 2:
                child = self.crossover(current_pop[0], current_pop[1])
                new_variants.append(child)
                
            # Evaluate all new variants
            scored_pop = await self.evaluate_fitness(new_variants, target_audience)
            
            # Environmental Selection (Survival of the fittest)
            current_pop = scored_pop[:2] 
            
            logger.info(f"🏆 Gen {gen} Top Score: {current_pop[0].fitness_score}")
            
        return current_pop[0]
