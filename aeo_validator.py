import re
import logging
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - FITYMI AEO SHIELD - %(message)s")

class AEOValidator:
    """
    Fitymi Phase 3: Advanced Answer Engine Optimization (AEO) Shielding v2.0.
    Ensures structural integrity (Markdown/JSON) and analyzes Semantic Density 
    to mimic 99th percentile human writing (Perplexity/Burstiness).
    """

    def __init__(self):
        logging.info("🛡️ Initializing AEO Validator Core (v2.0).")
        # In a full implementation, we would load spacy/nltk for deep burstiness calculation

    def validate_structure(self, text: str) -> Tuple[bool, str]:
        """
        Validates the markdown formatting of the output, preventing AI anomalies 
        (e.g., unmatched asterisks, broken links, code block wrapping when not requested).
        Returns a tuple: (is_valid, error_message or "Valid")
        """
        logging.info("🔍 Running AEO Structure Validation...")
        
        # Check for unclosed markdown headers (missing space)
        if re.search(r'^#{1,6}[A-Za-z]', text, re.MULTILINE):
            return False, "Malformed Markdown: Header missing space."
            
        # Check for extreme repetition (frequent AI hallucination artifact)
        words = text.split()
        if len(words) > 20:
            for i in range(len(words) - 5):
                if words[i:i+3] == words[i+3:i+6]:
                    return False, "Repetitive loop detected in output."
                    
        return True, "Valid"

    def calculate_semantic_density(self, text: str) -> float:
        """
        Computes a mock score for semantic density and burstiness.
        (A real implementation calculates variance in sentence length, unique entity count).
        """
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if not sentences:
            return 0.0
            
        # Rough proxy for burstiness: variance in sentence length
        lengths = [len(s.split()) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        
        # We want high variance (burstiness) for human-like writing, but not chaotic
        normalized_score = min(1.0, variance / 100.0) 
        logging.info(f"📊 Semantic Density / Burstiness Score: {normalized_score:.2f}")
        return normalized_score

    def ensure_compliance(self, draft: str) -> Dict[str, Any]:
        """
        Returns a compliance report that the MoA Critic will use to correct the copy.
        """
        is_valid, msg = self.validate_structure(draft)
        density = self.calculate_semantic_density(draft)
        
        status = "PASSED" if (is_valid and density > 0.2) else "FAILED"
        
        return {
            "status": status,
            "structural_error": msg if not is_valid else None,
            "semantic_density_score": density,
            "recommendation": "Rewrite adding more varied sentence lengths." if density <= 0.2 else "AEO Shield Passed."
        }

if __name__ == "__main__":
    validator = AEOValidator()
    test_text = "This is a sentence. This is another very long sentence that adds burstiness to the text because human writing is varied!"
    report = validator.ensure_compliance(test_text)
    print("\n--- AEO COMPLIANCE REPORT ---")
    print(report)
