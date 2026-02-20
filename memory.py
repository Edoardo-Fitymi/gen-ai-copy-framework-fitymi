import logging
import json
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - FITYMI MEMORY - %(message)s")

class MemoryNode:
    """Represents a discrete piece of knowledge in the GraphRAG Vector DB."""
    def __init__(self, node_id: str, content: str, metadata: Dict[str, Any], embedding: Optional[List[float]] = None):
        self.node_id = node_id
        self.content = content
        self.metadata = metadata
        self.embedding = embedding

class NexusMemoryCore:
    """
    Fitymi Phase 2: Dynamic RAG & Long-Term Brand Memory.
    This module simulates the integration with a Vector DB (like Pinecone/Chroma) and a Knowledge Graph 
    to retrieve Tone of Voice, past high-converting angles, and explicit brand constraints dynamically.
    """

    def __init__(self, db_provider: str = "chromadb"):
        self.db_provider = db_provider
        self.vector_store: List[MemoryNode] = []
        logging.info(f"🧠 Initializing Nexus Memory Core with {self.db_provider} backend.")
        self._bootstrap_memory()

    def _bootstrap_memory(self):
        """Populates the database with some initial enterprise brand knowledge."""
        self.vector_store.extend([
            MemoryNode(
                node_id="tov_001",
                content="Our brand voice is authoritative but empathetic. We do not use jargon unless necessary. We avoid words like '혁신적인' (innovative).",
                metadata={"type": "tone_of_voice", "brand": "TechCorp"}
            ),
            MemoryNode(
                node_id="angle_001",
                content="In Q3 2025, the 'Loss Aversion' angle out-performed the 'Aspirational' angle by 45% in our SaaS target.",
                metadata={"type": "historical_performance", "brand": "TechCorp", "target": "CTOs"}
            ),
            MemoryNode(
                node_id="product_001",
                content="Core Feature: Automated Security Posture Management. It saves approximately 20 hours per week for security teams.",
                metadata={"type": "product_knowledge", "brand": "TechCorp"}
            )
        ])

    def retrieve_context(self, brand: str, target: str, query: str = "") -> str:
        """
        Retrieves top-k relevant nodes from the Vector DB / GraphRAG based on semantic similarity to the query.
        (Using a deterministic mock retrieval for this Phase 2 architectural test).
        """
        logging.info(f"🔍 Retrieving GraphRAG context for Brand: {brand} | Target: {target}")
        
        # In a real implementation, we would embed the query and compute cosine similarity.
        # Here we mock the semantic matching.
        relevant_nodes = [
            node for node in self.vector_store 
            if node.metadata.get("brand") == brand
        ]
        
        context_blocks = []
        for node in relevant_nodes:
            context_blocks.append(f"[{node.metadata['type'].upper()}]: {node.content}")
            
        compiled_context = "\n".join(context_blocks)
        logging.info(f"📚 Context Retrieved: {len(relevant_nodes)} nodes")
        return compiled_context

    def update_learning(self, campaign_id: str, success_score: float, angle_used: str):
        """
        Phase 2 Continuous Learning: Stores the outcome of a copy strategy so the MoA can learn.
        """
        new_node = MemoryNode(
            node_id=f"learning_{campaign_id}",
            content=f"Campaign {campaign_id} using '{angle_used}' achieved a success score of {success_score}/1.0.",
            metadata={"type": "historical_performance", "score": success_score}
        )
        self.vector_store.append(new_node)
        logging.info(f"📈 Memory Updated. Continuous learning reinforced with score: {success_score}")
        
if __name__ == "__main__":
    memory = NexusMemoryCore()
    ctx = memory.retrieve_context(brand="TechCorp", target="CTOs")
    print("\n--- INJECTED CONTEXT ---")
    print(ctx)
