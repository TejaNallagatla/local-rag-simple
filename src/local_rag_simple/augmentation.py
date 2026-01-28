"""
COMPONENT 4: AUGMENTATION
Requirement: Combine retrieved info with query to enrich input
Tool: Custom Python string formatting
"""

class Augmentation:
    """
    Enriches query with retrieved context.
    Prepares structured input for generation.
    """
    
    def __init__(self):
        print("⚡ COMPONENT 4: Augmentation initialized")
    
    def create_context(self, query, retrieved_results):
        """
        Combine query with retrieved PDF chunks.
        
        This creates enriched input that includes:
        - Original user question
        - Relevant context from PDF
        - Metadata (page number)
        
        Args:
            query: User's question
            retrieved_results: List from retrieval system
        
        Returns:
            Enriched context string
        """
        print(f"\n⚡ Creating enriched context with {len(retrieved_results)} documents...")
        
        # Build context from retrieved docs
        context_parts = []
        for result in retrieved_results:
            doc = result['document']
            context_parts.append(
                f"[Page {doc['page']}]\n"
                f"Relevance: {result['similarity']:.2%}\n"
                f"{doc.get('chunk', '')}\n"
            )
        
        # Combine everything
        full_context = "\n---\n".join(context_parts)
        
        # Create enriched prompt
        enriched = f"""QUESTION: {query}

RELEVANT CONTEXT FROM PDF:
{full_context}

INSTRUCTIONS: Answer the question using the context above."""
        
        print(f"✅ Enriched context created ({len(enriched)} characters)")
        return enriched