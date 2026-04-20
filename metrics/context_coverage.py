"""
Context Recall — measures how much of the ground truth is covered by retrieved context.
Context Precision — measures what fraction of retrieved context is actually relevant.
"""
import numpy as np
from typing import List
from core.base_metric import BaseMetric
from core.models import RAGSample, MetricResult, MetricName


class ContextRecallMetric(BaseMetric):
    """
    Context Recall measures whether the retrieved context contains
    the information needed to answer the question correctly.

    Score = fraction of ground truth sentences attributable to context.
    """

    def __init__(self, threshold: float = 0.6, use_llm: bool = True):
        super().__init__(threshold)
        self.use_llm = use_llm
        self._client = None

    @property
    def name(self) -> MetricName:
        return MetricName.CONTEXT_RECALL

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

    def _check_ground_truth_attribution(
        self, ground_truth: str, contexts: List[str]
    ) -> tuple[float, List[dict]]:
        """Check which parts of ground truth can be attributed to context."""
        client = self._get_client()
        context_text = "\n\n".join(
            f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(contexts)
        )
        prompt = f"""For each sentence in the ground truth answer, determine if it can be attributed to the provided context.

Context:
{context_text}

Ground Truth Answer: {ground_truth}

For each sentence in the ground truth, respond with:
ATTRIBUTED: <sentence> | Reason: <brief reason>
NOT_ATTRIBUTED: <sentence> | Reason: <brief reason>

List each sentence separately."""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text
        
        attributed = []
        not_attributed = []
        
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("ATTRIBUTED:"):
                parts = line.replace("ATTRIBUTED:", "").split("|")
                sentence = parts[0].strip()
                reason = parts[1].replace("Reason:", "").strip() if len(parts) > 1 else ""
                attributed.append({"sentence": sentence, "reason": reason})
            elif line.startswith("NOT_ATTRIBUTED:"):
                parts = line.replace("NOT_ATTRIBUTED:", "").split("|")
                sentence = parts[0].strip()
                reason = parts[1].replace("Reason:", "").strip() if len(parts) > 1 else ""
                not_attributed.append({"sentence": sentence, "reason": reason})

        total = len(attributed) + len(not_attributed)
        if total == 0:
            return 0.5, []  # Unknown

        score = len(attributed) / total
        all_sentences = (
            [{"attributed": True, **a} for a in attributed] +
            [{"attributed": False, **n} for n in not_attributed]
        )
        return score, all_sentences

    def _simple_recall(self, ground_truth: str, contexts: List[str]) -> float:
        """Simple token overlap recall."""
        import re
        gt_tokens = set(re.findall(r'\b\w+\b', ground_truth.lower()))
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                     "at", "to", "for", "of", "and", "or", "it", "this", "that"}
        gt_tokens -= stopwords

        ctx_tokens = set()
        for ctx in contexts:
            ctx_tokens.update(re.findall(r'\b\w+\b', ctx.lower()))
        ctx_tokens -= stopwords

        if not gt_tokens:
            return 0.0
        return len(gt_tokens & ctx_tokens) / len(gt_tokens)

    def compute(self, sample: RAGSample) -> MetricResult:
        if not sample.ground_truth or not sample.contexts:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                explanation="Missing ground truth or contexts",
                threshold=self.threshold
            )

        if self.use_llm:
            try:
                score, sentence_details = self._check_ground_truth_attribution(
                    sample.ground_truth, sample.contexts
                )
                explanation = (
                    f"{score:.1%} of ground truth attributable to context. "
                    f"{sum(1 for s in sentence_details if s.get('attributed', False))} "
                    f"of {len(sentence_details)} sentences found in context."
                )
                details = {"sentence_attribution": sentence_details}
            except Exception:
                score = self._simple_recall(sample.ground_truth, sample.contexts)
                explanation = f"Token recall: {score:.1%} of ground truth terms in context"
                details = {"method": "token_overlap"}
        else:
            score = self._simple_recall(sample.ground_truth, sample.contexts)
            explanation = f"Token recall: {score:.1%} of ground truth terms in context"
            details = {"method": "token_overlap"}

        return MetricResult(
            metric_name=self.name,
            score=round(score, 4),
            explanation=explanation,
            details=details,
            threshold=self.threshold
        )


class ContextPrecisionMetric(BaseMetric):
    """
    Context Precision measures the signal-to-noise ratio of retrieved contexts.
    
    Score = fraction of retrieved chunks that are actually relevant to the question.
    """

    def __init__(self, threshold: float = 0.6, use_llm: bool = True):
        super().__init__(threshold)
        self.use_llm = use_llm
        self._client = None

    @property
    def name(self) -> MetricName:
        return MetricName.CONTEXT_PRECISION

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

    def _evaluate_context_relevance(
        self, question: str, contexts: List[str], ground_truth: str
    ) -> List[dict]:
        """Evaluate relevance of each context chunk."""
        client = self._get_client()
        results = []

        for i, ctx in enumerate(contexts):
            prompt = f"""Is the following context chunk relevant for answering the question?

Question: {question}
Expected Answer: {ground_truth}

Context chunk: {ctx[:500]}

Answer with RELEVANT or IRRELEVANT, then explain in one sentence.
Format: RELEVANT/IRRELEVANT: reason"""

            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text.strip()
                relevant = text.upper().startswith("RELEVANT")
                reason = text.split(":", 1)[-1].strip() if ":" in text else ""
                results.append({
                    "context_index": i,
                    "relevant": relevant,
                    "reason": reason,
                    "context_preview": ctx[:100] + "..."
                })
            except Exception:
                # Fallback: keyword match
                q_words = set(question.lower().split())
                c_words = set(ctx.lower().split())
                overlap = len(q_words & c_words) / max(len(q_words), 1)
                results.append({
                    "context_index": i,
                    "relevant": overlap > 0.2,
                    "reason": f"Keyword overlap: {overlap:.2f}",
                    "context_preview": ctx[:100] + "..."
                })
        return results

    def compute(self, sample: RAGSample) -> MetricResult:
        if not sample.contexts or not sample.question:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                explanation="Missing contexts or question",
                threshold=self.threshold
            )

        if self.use_llm:
            context_results = self._evaluate_context_relevance(
                sample.question, sample.contexts, sample.ground_truth
            )
        else:
            # Keyword fallback
            q_words = set(sample.question.lower().split())
            context_results = [
                {
                    "context_index": i,
                    "relevant": len(q_words & set(ctx.lower().split())) / max(len(q_words), 1) > 0.15,
                    "reason": "keyword overlap",
                    "context_preview": ctx[:100]
                }
                for i, ctx in enumerate(sample.contexts)
            ]

        # Weighted precision — earlier contexts get higher weight
        n = len(context_results)
        weighted_precision = 0.0
        cumulative_relevant = 0

        for rank, result in enumerate(context_results, 1):
            if result["relevant"]:
                cumulative_relevant += 1
                precision_at_k = cumulative_relevant / rank
                weighted_precision += precision_at_k

        if cumulative_relevant > 0:
            score = weighted_precision / cumulative_relevant
        else:
            score = 0.0

        relevant_count = sum(1 for r in context_results if r["relevant"])
        explanation = (
            f"{relevant_count}/{n} retrieved contexts are relevant. "
            f"Weighted precision@k: {score:.3f}"
        )

        return MetricResult(
            metric_name=self.name,
            score=round(score, 4),
            explanation=explanation,
            details={"context_relevance": context_results},
            threshold=self.threshold
        )
