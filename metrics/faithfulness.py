"""
Faithfulness Metric — measures whether all claims in the answer
are supported by the retrieved context (no hallucinations).

Score: fraction of answer claims supported by context.
"""
import os
import re
from typing import List
from core.base_metric import BaseMetric
from core.models import RAGSample, MetricResult, MetricName


class FaithfulnessMetric(BaseMetric):
    """
    Faithfulness evaluates whether the generated answer contains only
    information that is supported by the retrieved context.

    High faithfulness = no hallucinations.
    Score = supported_claims / total_claims
    """

    def __init__(self, threshold: float = 0.7, use_llm: bool = True):
        super().__init__(threshold)
        self.use_llm = use_llm
        self._client = None

    @property
    def name(self) -> MetricName:
        return MetricName.FAITHFULNESS

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

    def _extract_claims_llm(self, answer: str) -> List[str]:
        """Use LLM to extract atomic claims from the answer."""
        client = self._get_client()
        prompt = f"""Extract all atomic factual claims from this answer. 
Return each claim on a new line starting with '- '.
Only include statements that assert facts, not questions or opinions.

Answer: {answer}

Claims:"""
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text
        claims = [
            line.lstrip("- ").strip()
            for line in text.split("\n")
            if line.strip().startswith("-")
        ]
        return claims if claims else [answer]

    def _extract_claims_simple(self, answer: str) -> List[str]:
        """Simple sentence-based claim extraction fallback."""
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _check_claim_supported_llm(self, claim: str, contexts: List[str]) -> tuple[bool, str]:
        """Use LLM to check if a claim is supported by the context."""
        client = self._get_client()
        context_text = "\n\n".join(f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(contexts))
        prompt = f"""Determine if the following claim is fully supported by the provided context.

Context:
{context_text}

Claim: {claim}

Answer with exactly one of:
SUPPORTED - the claim is directly stated or clearly implied by the context
NOT SUPPORTED - the claim contradicts or is not found in the context

Then briefly explain why (1 sentence).

Format: SUPPORTED/NOT SUPPORTED: explanation"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        supported = text.upper().startswith("SUPPORTED")
        explanation = text.split(":", 1)[-1].strip() if ":" in text else text
        return supported, explanation

    def _check_claim_supported_simple(self, claim: str, contexts: List[str]) -> tuple[bool, str]:
        """Simple keyword overlap check as fallback."""
        claim_words = set(claim.lower().split())
        # Remove stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "and", "or", "but", "in", "on", "at", "to", "for", "of", "it"}
        claim_words -= stopwords

        for ctx in contexts:
            ctx_words = set(ctx.lower().split()) - stopwords
            overlap = len(claim_words & ctx_words)
            if overlap / max(len(claim_words), 1) > 0.4:
                return True, "Keyword overlap found in context"
        return False, "No supporting context found"

    def compute(self, sample: RAGSample) -> MetricResult:
        if not sample.answer:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                explanation="No answer provided",
                threshold=self.threshold
            )

        # Step 1: Extract claims
        if self.use_llm:
            try:
                claims = self._extract_claims_llm(sample.answer)
            except Exception:
                claims = self._extract_claims_simple(sample.answer)
        else:
            claims = self._extract_claims_simple(sample.answer)

        if not claims:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                explanation="No factual claims to verify",
                threshold=self.threshold
            )

        # Step 2: Check each claim against context
        supported_count = 0
        claim_details = []

        for claim in claims:
            if self.use_llm:
                try:
                    supported, explanation = self._check_claim_supported_llm(
                        claim, sample.contexts
                    )
                except Exception:
                    supported, explanation = self._check_claim_supported_simple(
                        claim, sample.contexts
                    )
            else:
                supported, explanation = self._check_claim_supported_simple(
                    claim, sample.contexts
                )

            if supported:
                supported_count += 1
            claim_details.append({
                "claim": claim,
                "supported": supported,
                "explanation": explanation
            })

        score = supported_count / len(claims)
        unsupported = [c for c in claim_details if not c["supported"]]

        explanation = (
            f"{supported_count}/{len(claims)} claims supported by context."
        )
        if unsupported:
            explanation += f" Unsupported: {unsupported[0]['claim'][:80]}..."

        return MetricResult(
            metric_name=self.name,
            score=round(score, 4),
            explanation=explanation,
            details={
                "total_claims": len(claims),
                "supported_claims": supported_count,
                "claim_details": claim_details
            },
            threshold=self.threshold
        )
