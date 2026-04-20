"""
Metrics Registry — central factory for all evaluation metrics.
"""
from typing import List
from core.models import MetricName, BenchmarkConfig
from core.base_metric import BaseMetric
from metrics.fact_accuracy import FaithfulnessMetric
from metrics.answer_quality import AnswerRelevanceMetric
from metrics.context_coverage import ContextRecallMetric, ContextPrecisionMetric
from metrics.document_grounding import GroundednessMetric
from metrics.semantic_similarity import SemanticSimilarityMetric, ROUGELMetric


METRIC_DESCRIPTIONS = {
    MetricName.FAITHFULNESS:        "Fact Accuracy — checks that every claim in the answer is actually supported by the retrieved document. High score = no made-up facts.",
    MetricName.ANSWER_RELEVANCE:    "Answers the Question — measures how well the answer addresses what was actually asked. High score = on-topic, not a tangent.",
    MetricName.CONTEXT_RECALL:      "Info Coverage — checks whether the retrieved chunks contain all the information needed to answer correctly. High score = nothing important was missed.",
    MetricName.CONTEXT_PRECISION:   "Source Quality — measures whether the retrieved chunks are genuinely relevant. High score = no noisy or off-topic chunks polluting the context.",
    MetricName.GROUNDEDNESS:        "Stays in Document — overall judge score checking the answer does not go beyond what the document says. High score = well-grounded in source.",
    MetricName.SEMANTIC_SIMILARITY: "Meaning Match — how close in meaning the answer is to the expected correct answer, even if worded differently. High score = same idea expressed.",
    MetricName.ROUGE_L:             "Word Overlap — counts matching words/phrases between the answer and the expected answer. High score = similar wording to ground truth.",
}


def build_metrics(config: BenchmarkConfig) -> List[BaseMetric]:
    """Build a list of metric instances from a BenchmarkConfig."""
    metrics = []
    use_llm = config.use_llm_judge
    embedding_model = config.embedding_model

    metric_map = {
        MetricName.FAITHFULNESS: lambda: FaithfulnessMetric(
            threshold=config.thresholds.get("faithfulness", 0.7),
            use_llm=use_llm
        ),
        MetricName.ANSWER_RELEVANCE: lambda: AnswerRelevanceMetric(
            threshold=config.thresholds.get("answer_relevance", 0.7),
            use_llm=use_llm,
            embedding_model=embedding_model
        ),
        MetricName.CONTEXT_RECALL: lambda: ContextRecallMetric(
            threshold=config.thresholds.get("context_recall", 0.6),
            use_llm=use_llm
        ),
        MetricName.CONTEXT_PRECISION: lambda: ContextPrecisionMetric(
            threshold=config.thresholds.get("context_precision", 0.6),
            use_llm=use_llm
        ),
        MetricName.GROUNDEDNESS: lambda: GroundednessMetric(
            threshold=config.thresholds.get("groundedness", 0.7),
            use_llm=use_llm
        ),
        MetricName.SEMANTIC_SIMILARITY: lambda: SemanticSimilarityMetric(
            threshold=config.thresholds.get("semantic_similarity", 0.75),
            embedding_model=embedding_model
        ),
        MetricName.ROUGE_L: lambda: ROUGELMetric(
            threshold=config.thresholds.get("rouge_l", 0.3)
        ),
    }

    for metric_name in config.metrics:
        if metric_name in metric_map:
            metrics.append(metric_map[metric_name]())

    return metrics
