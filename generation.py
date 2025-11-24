"""
COMPONENT 5: GENERATION
Requirement: Generate response grounded in retrieved knowledge
Tool: Template-based synthesis (no LLM needed for demo)
"""

class Generation:
    """
    Generates final answer from enriched context.
    Uses simple template-based approach (can be replaced with LLM).
    """
    
    def __init__(self):
        print("🤖 COMPONENT 5: Generation initialized (simple mode)")
    
    def generate(self, enriched_context, retrieved_results):
        """
        Generate answer from enriched context.
        
        For this demo: formats retrieved context clearly.
        In production: would use LLM like GPT/Claude/Llama.
        
        Args:
            enriched_context: Output from augmentation
            retrieved_results: Original retrieval results
        
        Returns:
            Generated answer string
        """
        print("\n🤖 Generating answer...")
        
        # Extract question
        question_line = enriched_context.split('\n')[0]
        
        # Format answer from retrieved docs
        answer_parts = [f"Based on {len(retrieved_results)} relevant scenes from pdf:\n"]
        
        for i, result in enumerate(retrieved_results, 1):
            doc = result['document']
            preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
            
            answer_parts.append(
                f"\n{i}. Page {doc['page']} (relevance: {result['similarity']:.1%})\n"
                f"   {preview}\n"
            )
        
        answer = "".join(answer_parts)
        
        print("✅ Answer generated")
        return answer