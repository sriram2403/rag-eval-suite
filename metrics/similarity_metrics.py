"""
Semantic Similarity & ROUGE-L Metrics
- Semantic Similarity: embedding cosine similarity between answer and ground truth
- ROUGE-L: longest common subsequence based recall
"""
import numpy as np
from core.base_metric import BaseMetric
from core.models import RAGSample, MetricResult, MetricName


class SemanticSimilarityMetric(BaseMetric):
    """Cosine similarity between answer and ground truth embeddings."""

    def __init__(self, threshold: float = 0.75, embedding_model: str = "all-MiniLM-L6-v2"):
        super().__init__(threshold)
        self.embedding_model_name = embedding_model
        self._embedder = None

    @property
    def name(self) -> MetricName:
        return MetricName.SEMANTIC_SIMILARITY

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def compute(self, sample: RAGSample) -> MetricResult:
        if not sample.answer or not sample.ground_truth:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                explanation="Missing answer or ground truth",
                threshold=self.threshold
            )

        try:
            embedder = self._get_embedder()
            emb_answer = embedder.encode(sample.answer)
            emb_truth = embedder.encode(sample.ground_truth)

            # Cosine similarity
            norm_a = np.linalg.norm(emb_answer)
            norm_b = np.linalg.norm(emb_truth)
            if norm_a == 0 or norm_b == 0:
                score = 0.0
            else:
                score = float(np.dot(emb_answer, emb_truth) / (norm_a * norm_b))

            # Normalize from [-1,1] to [0,1]
            score = (score + 1) / 2

        except Exception:
            # Jaccard fallback
            a_words = set(sample.answer.lower().split())
            b_words = set(sample.ground_truth.lower().split())
            score = len(a_words & b_words) / max(len(a_words | b_words), 1)

        explanation = f"Semantic similarity between answer and ground truth: {score:.3f}"

        return MetricResult(
            metric_name=self.name,
            score=round(score, 4),
            explanation=explanation,
            details={"embedding_model": self.embedding_model_name},
            threshold=self.threshold
        )


class ROUGELMetric(BaseMetric):
    """ROUGE-L: longest common subsequence F1 between answer and ground truth."""

    def __init__(self, threshold: float = 0.3):
        super().__init__(threshold)

    @property
    def name(self) -> MetricName:
        return MetricName.ROUGE_L

    def _lcs_length(self, x: list, y: list) -> int:
        """Compute LCS length using dynamic programming."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    def compute(self, sample: RAGSample) -> MetricResult:
        if not sample.answer or not sample.ground_truth:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                explanation="Missing answer or ground truth",
                threshold=self.threshold
            )

        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(sample.ground_truth, sample.answer)
            score = scores['rougeL'].fmeasure
        except ImportError:
            # Manual LCS implementation
            ref_tokens = sample.ground_truth.lower().split()
            hyp_tokens = sample.answer.lower().split()

            lcs = self._lcs_length(ref_tokens, hyp_tokens)
            precision = lcs / max(len(hyp_tokens), 1)
            recall = lcs / max(len(ref_tokens), 1)

            if precision + recall > 0:
                score = 2 * precision * recall / (precision + recall)
            else:
                score = 0.0

        explanation = f"ROUGE-L F1 score: {score:.3f} (longest common subsequence overlap)"

        return MetricResult(
            metric_name=self.name,
            score=round(score, 4),
            explanation=explanation,
            threshold=self.threshold
        )
