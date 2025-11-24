"""
COMPONENT 5: GENERATION
Requirement: Generate response grounded in retrieved knowledge
Tool: Ollama (local LLM)
"""

import ollama

class Generation:
    """
    Generates final answer from enriched context using Ollama LLM.
    """
    
    def __init__(self, model_name='llama3.2:3b', use_llm=True):
        """
        Initialize generation component.
        
        Args:
            model_name: Ollama model to use (default: llama3.2:3b)
            use_llm: If False, falls back to template-based generation
        """
        self.model_name = model_name
        self.use_llm = use_llm
        
        if use_llm:
            print(f"🤖 COMPONENT 5: Generation initialized (Ollama: {model_name})")
            
            # Test Ollama connection
            try:
                ollama.list()
                print("✅ Ollama connection successful")
            except Exception as e:
                print(f"⚠️ Warning: Could not connect to Ollama: {e}")
                print("   Make sure Ollama is running: brew services start ollama")
                print("   Falling back to template mode...")
                self.use_llm = False
        else:
            print("🤖 COMPONENT 5: Generation initialized (template mode)")
    
    def generate(self, enriched_context, retrieved_results):
        """
        Generate answer from enriched context.
        
        Args:
            enriched_context: Full prompt with question and context
            retrieved_results: Original retrieval results (for metadata)
        
        Returns:
            Generated answer string
        """
        if self.use_llm:
            return self._generate_with_llm(enriched_context, retrieved_results)
        else:
            return self._generate_template(enriched_context, retrieved_results)
    
    def _generate_with_llm(self, enriched_context, retrieved_results):
        """Generate answer using Ollama LLM."""
        print("\n🤖 Generating answer with LLM...")
        
        try:
            # Call Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': enriched_context
                    }
                ],
                options={
                    'temperature': 0.7,      # Creativity (0=focused, 1=creative)
                    'num_predict': 500,      # Max tokens in response
                    'top_k': 40,             # Limits vocabulary consideration
                    'top_p': 0.9,            # Nucleus sampling
                }
            )
            
            # Extract answer from response
            answer = response['message']['content']
            
            # Add source citations at the end
            answer += "\n\n" + "="*70
            answer += "\n📚 SOURCES:\n"
            answer += "="*70 + "\n"
            
            for i, result in enumerate(retrieved_results, 1):
                doc = result['document']
                answer += f"\n[{i}] Page {doc['page']} (Similarity: {result['similarity']:.1%})\n"
                preview = doc['chunk'][:150] + "..." if len(doc['chunk']) > 150 else doc['chunk']
                answer += f"    Preview: {preview}\n"
            
            print("✅ Answer generated successfully with LLM")
            return answer
            
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            print("   Falling back to template mode...")
            return self._generate_template(enriched_context, retrieved_results)
    
    def _generate_template(self, enriched_context, retrieved_results):
        """
        Fallback template-based answer (when LLM unavailable).
        """
        print("\n🤖 Generating answer (template mode)...")
        
        # Extract question
        lines = enriched_context.split('\n')
        question_line = lines[0] if lines else "Question not found"
        
        # Build answer
        answer_parts = [
            "="*70,
            "\n⚠️  LLM Mode Disabled - Showing Retrieved Context",
            "\n" + "="*70,
            f"\n\n{question_line}\n",
            f"\nFound {len(retrieved_results)} relevant passages:\n\n"
        ]
        
        for i, result in enumerate(retrieved_results, 1):
            doc = result['document']
            preview = doc['chunk'][:300] + "..." if len(doc['chunk']) > 300 else doc['chunk']
            
            answer_parts.append(
                f"[{i}] Page {doc['page']} (Similarity: {result['similarity']:.1%})\n"
                f"{preview}\n\n"
            )
        
        answer_parts.append("="*70)
        answer_parts.append("\n💡 TIP: Enable LLM mode for AI-generated answers")
        answer_parts.append("\n" + "="*70)
        
        answer = "".join(answer_parts)
        
        print("✅ Answer generated (template mode)")
        return answer