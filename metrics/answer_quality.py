"""
Answer Relevance Metric — measures whether the generated answer
actually addresses the question asked.

Score: semantic similarity between generated question-from-answer and original question.
"""
import numpy as np
from typing import List
from core.base_metric import BaseMetric
from core.models import RAGSample, MetricResult, MetricName


class AnswerRelevanceMetric(BaseMetric):
    """
    Answer Relevance measures if the answer addresses the question.
    
    Method:
    1. Use LLM to generate N questions from the answer
    2. Compute cosine similarity between generated questions and original question
    3. Score = mean similarity (high = answer addresses the question)
    """

    def __init__(self, threshold: float = 0.7, use_llm: bool = True,
                 embedding_model: str = "all-MiniLM-L6-v2", n_questions: int = 3):
        super().__init__(threshold)
        self.use_llm = use_llm
        self.embedding_model_name = embedding_model
        self.n_questions = n_questions
        self._embedder = None
        self._client = None

    @property
    def name(self) -> MetricName:
        return MetricName.ANSWER_RELEVANCE

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _get_client(self):
        if self._client is None:
            import os
            if os.environ.get("GROQ_API_KEY"):
                from groq_client import GroqClient
                self._client = GroqClient()
            else:
                import anthropic
                self._client = anthropic.Anthropic()
        return self._client

    def _generate_questions_from_answer(self, answer: str) -> List[str]:
        """Generate questions that the answer could be responding to."""
        client = self._get_client()
        prompt = f"""Given this answer, generate {self.n_questions} different questions that this answer could be responding to.
Each question should be on its own line starting with 'Q: '.

Answer: {answer}

Questions:"""
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text
        questions = [
            line.replace("Q:", "").strip()
            for line in text.split("\n")
            if line.strip().startswith("Q:")
        ]
        return questions if questions else [answer[:100]]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def compute(self, sample: RAGSample) -> MetricResult:
        if not sample.answer or not sample.question:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                explanation="Missing answer or question",
                threshold=self.threshold
            )

        # Generate questions from the answer
        if self.use_llm:
            try:
                generated_questions = self._generate_questions_from_answer(sample.answer)
            except Exception:
                generated_questions = [sample.answer[:100]]
        else:
            # Fallback: use answer as proxy
            generated_questions = [sample.answer[:200]]

        # Embed original question and generated questions
        try:
            embedder = self._get_embedder()
            original_embedding = embedder.encode(sample.question)
            generated_embeddings = embedder.encode(generated_questions)

            similarities = [
                self._cosine_similarity(original_embedding, gen_emb)
                for gen_emb in generated_embeddings
            ]
            score = float(np.mean(similarities))
        except Exception as e:
            # Keyword fallback
            q_words = set(sample.question.lower().split())
            a_words = set(sample.answer.lower().split())
            overlap = len(q_words & a_words) / max(len(q_words), 1)
            score = min(overlap * 2, 1.0)
            similarities = [score]
            generated_questions = []

        explanation = (
            f"Answer addresses the question with score {score:.3f}. "
            f"Generated {len(generated_questions)} reverse questions, "
            f"mean similarity: {score:.3f}"
        )

        return MetricResult(
            metric_name=self.name,
            score=round(score, 4),
            explanation=explanation,
            details={
                "generated_questions": generated_questions,
                "similarities": [round(s, 4) for s in similarities],
                "mean_similarity": round(score, 4)
            },
            threshold=self.threshold
        )
