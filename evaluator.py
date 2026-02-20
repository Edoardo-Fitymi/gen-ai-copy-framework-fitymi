import logging
from agent import FitymiCopyAgent, FitymiPayload

logging.basicConfig(level=logging.INFO, format="%(asctime)s - FITYMI JUDGE - %(message)s")

class AutonomousEvaluator:
    """
    Fitymi Phase 4: Autonomous Evaluation (LLM-as-a-Judge) and RLAIF.
    Simulates A/B testing by exposing the copy to synthetic personas and running adversarial tests.
    If the score is below the threshold, it triggers a recursive rewrite.
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        logging.info(f"⚖️ Initializing LLM-as-a-Judge ({model}).")
        self.judge_agent = FitymiCopyAgent(provider=provider, model=model)

    async def evaluate_copy(self, draft: str, target_audience: str, goal: str) -> float:
        """
        Runs synthetic A/B testing on the copy. 
        Returns a probability of success score between 0.0 and 1.0.
        """
        logging.info("🧪 Running Synthetic A/B Testing & Adversarial Evaluation...")
        
        prompt = f"""
        You are a highly analytical AI simulating the target audience: {target_audience}.
        Your goal is to be extremely skeptical of marketing copy. 
        
        Evaluate the following draft against the goal: {goal}.
        Will this make you take action? Does it sound like AI or a real human?
        
        Draft:
        {draft}
        
        Output exclusively a single float number between 0.0 and 1.0 representing the probability of conversion.
        0.0 = Absolute trash, clear AI writing, no conversion.
        1.0 = Masterpiece, human-sounding, immediate conversion.
        Do not output any text other than the number.
        """
        
        payload = FitymiPayload(
            system_prompt="You are an uncompromising, skeptical marketing judge.",
            user_context=prompt,
            task_definition="Score the copy.",
            verification_protocol="Ensure only a float is outputted.",
            aeo_shielding="Provide the exact float number without markdown or extra text."
        )
        
        response = await self.judge_agent.execute(payload)
        
        try:
            score = float(response.raw_output.strip())
            logging.info(f"🏆 Autonomous Eval Score: {score}/1.0")
            return score
        except ValueError:
            logging.error("Failed to parse evaluation score. Defaulting to 0.5")
            return 0.5

if __name__ == "__main__":
    import asyncio
    async def test():
        evaluator = AutonomousEvaluator()
        test_draft = "Transform your workflow with our revolutionary AI solution today in the vast world of tech!"
        score = await evaluator.evaluate_copy(test_draft, "Skeptical CTOs", "Sign up for demo")
        print(f"Synthetic Score: {score}")

    asyncio.run(test())
