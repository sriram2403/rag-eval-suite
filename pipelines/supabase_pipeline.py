"""
Supabase RAG Pipeline — retrieves from vector DB, generates with Groq.
"""
import os
from typing import List
from pipelines.rag_pipelines import BasePipeline
from document_processor import search_documents_hybrid, fetch_all_chunks, keyword_rank


class SupabaseRAGPipeline(BasePipeline):
    """
    Production RAG pipeline using:
    - Supabase pgvector for retrieval
    - Groq/Llama for generation
    """

    def __init__(self, top_k: int = 6, filename: str = None, threshold: float = 0.2):
        self.top_k = top_k
        self.filename = filename  # filter to specific document
        self.threshold = threshold
        self._client = None

    @property
    def name(self) -> str:
        return "supabase_rag_pgvector"

    def _get_client(self):
        if self._client is None:
            if os.environ.get("GROQ_API_KEY"):
                from groq_client import GroqClient
                self._client = GroqClient()
            else:
                import anthropic
                self._client = anthropic.Anthropic()
        return self._client

    def retrieve(self, question: str) -> List[str]:
        """
        Retrieval strategy:
        1. Always do a full keyword scan over all stored chunks (reliable baseline)
        2. Also run hybrid vector+keyword search for semantic candidates
        3. Merge: keyword-top chunks first, fill remaining slots with semantic candidates
        This ensures correct retrieval even when vector embeddings are degraded.
        """
        try:
            # Step 1: full Python keyword scan — always runs, immune to bad embeddings
            all_chunks = fetch_all_chunks(filename=self.filename)
            keyword_top = keyword_rank(question, all_chunks, top_k=4) if all_chunks else []

            # Step 2: hybrid vector search for semantic diversity
            try:
                semantic = search_documents_hybrid(
                    query=question,
                    filename=self.filename,
                    top_k=self.top_k,
                    threshold=self.threshold
                )
            except Exception:
                semantic = []

            # Step 3: merge — keyword results take priority, fill to 4 with semantic
            seen = set()
            result = []
            for chunk in keyword_top + semantic:
                if chunk not in seen:
                    seen.add(chunk)
                    result.append(chunk)
                if len(result) == 4:
                    break

            return result if result else ["No relevant documents found."]

        except Exception as e:
            return [f"Search error: {e}"]

    def generate(self, question: str, contexts: List[str]) -> str:
        """
        3-stage generation:
        1. Answer from document
        2. Detect if insufficient
        3. Extend with reasoning if needed
        """
        client = self._get_client()

        context_text = "\n\n".join(
            f"[CHUNK {i+1}]:\n{ctx}" for i, ctx in enumerate(contexts)
        )

        # ── Stage 1: Extract what document actually says ──
        extraction_prompt = f"""You are reading document chunks to answer a question.

DOCUMENT CHUNKS:
{context_text}

QUESTION: {question}

Task: Extract ALL information from the chunks that is relevant to this question.
Be thorough — include formulas, definitions, implications, and related concepts.
If nothing is relevant, say "INSUFFICIENT".

EXTRACTED INFORMATION:"""

        extraction_response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": extraction_prompt}]
        )
        extracted = extraction_response.content[0].text.strip()

        # ── Stage 2: Check if document content is sufficient ──
        is_insufficient = (
            "INSUFFICIENT" in extracted.upper() or
            len(extracted.split()) < 30 or
            "does not contain" in extracted.lower() or
            "no information" in extracted.lower() or
            "not mentioned" in extracted.lower()
        )

        # ── Stage 3: Build final answer ──
        if not is_insufficient:
            answer_prompt = f"""Using this extracted information from the document,
write a clear complete answer to the question.

EXTRACTED FROM DOCUMENT:
{extracted}

QUESTION: {question}

Write a thorough answer using the document content.
If the document only partially answers the question, answer what it covers
and then reason beyond it — clearly marking the boundary.

FORMAT:
📄 FROM DOCUMENT:
[answer grounded in document]

💡 EXTENDED REASONING:
[your reasoning beyond the document — only if needed, skip if document is complete]

ANSWER:"""

        else:
            answer_prompt = f"""The document chunks provided have insufficient
information to fully answer this question.

WHAT THE DOCUMENT SAYS (partial):
{extracted}

QUESTION: {question}

Provide a complete answer by:
1. Using whatever the document does say (even if partial)
2. Reasoning from first principles and your knowledge for the rest
3. Clearly separating what came from the document vs your reasoning

FORMAT:
📄 FROM DOCUMENT:
[what the document actually contains — or "The document does not cover this directly"]

💡 EXTENDED REASONING:
[your full reasoning, derivations, explanations from knowledge]

ANSWER:"""

        final_response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[{"role": "user", "content": answer_prompt}]
        )

        return final_response.content[0].text.strip()

    def check_answer_groundedness(
        self, question: str, answer: str, contexts: List[str]
    ) -> tuple[str, float]:
        """
        Updated confidence check that accounts for reasoning extension.
        Answers with extended reasoning get partial credit.
        """
        if "FROM DOCUMENT" in answer and "EXTENDED REASONING" in answer:
            return answer, 0.7

        client = self._get_client()
        context_text = "\n\n".join(
            f"[CHUNK {i+1}]: {ctx}" for i, ctx in enumerate(contexts)
        )

        prompt = f"""Check if this answer is grounded in the context.

CONTEXT:
{context_text}

ANSWER: {answer}

Rate from 0.0 to 1.0 how grounded this answer is.
Respond with just a number like: 0.85"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            confidence = float(response.content[0].text.strip())
            confidence = max(0.0, min(1.0, confidence))
        except:
            confidence = 0.6

        return answer, confidence
    
    def run(self, question: str):
        """Full pipeline with document + reasoning fallback."""
        contexts = self.retrieve(question)
        answer = self.generate(question, contexts)

        # Check groundedness — hybrid answers get 0.7 automatically
        verified_answer, confidence = self.check_answer_groundedness(
            question, answer, contexts
        )

        # Only prepend warning for very low confidence pure answers
        if confidence < 0.3 and "FROM DOCUMENT" not in answer:
            verified_answer = (
                f"⚠️ Low document coverage ({confidence:.0%})\n\n{verified_answer}"
            )

        return verified_answer, contexts