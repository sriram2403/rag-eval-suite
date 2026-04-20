"""
Groundedness Metric — measures how well the answer is grounded
in the retrieved context (combines faithfulness + citation quality).

This is stricter than faithfulness: it checks not just if claims
are supported, but whether the answer stays within the bounds of
what the context actually says.
"""
from typing import List
from core.base_metric import BaseMetric
from core.models import RAGSample, MetricResult, MetricName


class GroundednessMetric(BaseMetric):
    """
    Groundedness is a holistic measure: does the answer only say
    things the context supports, and does it say them accurately?

    Scores:
    - 1.0: Fully grounded, all claims directly in context
    - 0.7-0.9: Mostly grounded with minor extrapolations
    - 0.4-0.7: Some hallucinations or unsupported claims
    - 0.0-0.4: Significantly ungrounded
    """

    def __init__(self, threshold: float = 0.7, use_llm: bool = True):
        super().__init__(threshold)
        self.use_llm = use_llm
        self._client = None

    @property
    def name(self) -> MetricName:
        return MetricName.GROUNDEDNESS

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

    def _llm_groundedness_score(
        self, question: str, answer: str, contexts: List[str]
    ) -> tuple[float, str, dict]:
        """Use LLM as judge to score groundedness holistically."""
        client = self._get_client()
        context_text = "\n\n".join(
            f"[Context {i+1}]:\n{ctx}" for i, ctx in enumerate(contexts)
        )

        prompt = f"""You are an expert evaluator assessing whether an AI answer is grounded in the provided context.

Question: {question}

Retrieved Context:
{context_text}

Generated Answer: {answer}

Evaluate the answer on groundedness using this rubric:
- FULLY_GROUNDED (0.9-1.0): Every claim is directly stated in the context
- MOSTLY_GROUNDED (0.7-0.89): Minor extrapolations, all key claims supported  
- PARTIALLY_GROUNDED (0.4-0.69): Some claims unsupported, noticeable hallucinations
- POORLY_GROUNDED (0.0-0.39): Many unsupported claims, significant hallucinations

Respond in this exact format:
LEVEL: <FULLY_GROUNDED/MOSTLY_GROUNDED/PARTIALLY_GROUNDED/POORLY_GROUNDED>
SCORE: <number between 0.0 and 1.0>
HALLUCINATIONS: <list any specific hallucinations or "None">
REASONING: <one paragraph explanation>"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()

        # Parse response
        score = 0.5
        level = "PARTIALLY_GROUNDED"
        hallucinations = []
        reasoning = ""

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score = float(line.replace("SCORE:", "").strip())
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    pass
            elif line.startswith("LEVEL:"):
                level = line.replace("LEVEL:", "").strip()
            elif line.startswith("HALLUCINATIONS:"):
                h_text = line.replace("HALLUCINATIONS:", "").strip()
                if h_text.lower() != "none":
                    hallucinations = [h.strip() for h in h_text.split(",")]
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return score, reasoning, {
            "level": level,
            "hallucinations": hallucinations,
            "reasoning": reasoning
        }

    def _simple_groundedness(self, answer: str, contexts: List[str]) -> float:
        """Simple overlap-based groundedness."""
        import re
        answer_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        ctx_words = set()
        for ctx in contexts:
            ctx_words.update(re.findall(r'\b\w{4,}\b', ctx.lower()))

        if not answer_words:
            return 0.0
        return len(answer_words & ctx_words) / len(answer_words)

    def compute(self, sample: RAGSample) -> MetricResult:
        if not sample.answer or not sample.contexts:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                explanation="Missing answer or context",
                threshold=self.threshold
            )

        if self.use_llm:
            try:
                score, reasoning, details = self._llm_groundedness_score(
                    sample.question, sample.answer, sample.contexts
                )
                explanation = reasoning or f"Groundedness score: {score:.3f}"
            except Exception:
                score = self._simple_groundedness(sample.answer, sample.contexts)
                explanation = f"Token-based groundedness: {score:.3f}"
                details = {"method": "token_overlap"}
        else:
            score = self._simple_groundedness(sample.answer, sample.contexts)
            explanation = f"Token-based groundedness: {score:.3f}"
            details = {"method": "token_overlap"}

        return MetricResult(
            metric_name=self.name,
            score=round(score, 4),
            explanation=explanation,
            details=details,
            threshold=self.threshold
        )
