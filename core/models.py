"""
Core data models for the RAG Evaluation Framework.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class MetricName(str, Enum):
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_RECALL = "context_recall"
    CONTEXT_PRECISION = "context_precision"
    GROUNDEDNESS = "groundedness"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    ROUGE_L = "rouge_l"


@dataclass
class RAGSample:
    """A single test sample for RAG evaluation."""
    question: str
    ground_truth: str
    contexts: List[str]
    answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    sample_id: str = ""

    def __post_init__(self):
        if not self.sample_id:
            self.sample_id = f"sample_{hash(self.question) % 100000:05d}"


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""
    metric_name: MetricName
    score: float                    # 0.0 to 1.0
    explanation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    passed: bool = False
    threshold: float = 0.7

    def __post_init__(self):
        self.passed = self.score >= self.threshold


@dataclass
class SampleEvalResult:
    """Full evaluation results for a single RAG sample."""
    sample: RAGSample
    metric_results: Dict[str, MetricResult] = field(default_factory=dict)
    overall_score: float = 0.0
    latency_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def compute_overall(self):
        if self.metric_results:
            self.overall_score = sum(
                r.score for r in self.metric_results.values()
            ) / len(self.metric_results)
        return self.overall_score


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    metrics: List[MetricName] = field(default_factory=lambda: [
        MetricName.FAITHFULNESS,
        MetricName.ANSWER_RELEVANCE,
        MetricName.CONTEXT_RECALL,
        MetricName.GROUNDEDNESS,
    ])
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "faithfulness": 0.7,
        "answer_relevance": 0.7,
        "context_recall": 0.6,
        "context_precision": 0.6,
        "groundedness": 0.7,
        "semantic_similarity": 0.75,
        "rouge_l": 0.3,
    })
    use_llm_judge: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    max_samples: Optional[int] = None
    verbose: bool = True


@dataclass
class BenchmarkReport:
    """Complete benchmark report across all samples."""
    config: BenchmarkConfig
    sample_results: List[SampleEvalResult] = field(default_factory=list)
    aggregate_scores: Dict[str, float] = field(default_factory=dict)
    pass_rates: Dict[str, float] = field(default_factory=dict)
    total_samples: int = 0
    passed_samples: int = 0
    run_duration_s: float = 0.0
    timestamp: str = ""
    pipeline_name: str = "unknown"

    def compute_aggregates(self):
        """Compute aggregate metrics across all samples."""
        if not self.sample_results:
            return

        metric_scores: Dict[str, List[float]] = {}
        for result in self.sample_results:
            for metric_name, metric_result in result.metric_results.items():
                # Normalize key to always be a plain string
                key = metric_name if isinstance(metric_name, str) else metric_name.value
                if key not in metric_scores:
                    metric_scores[key] = []
                metric_scores[key].append(metric_result.score)

        self.aggregate_scores = {
            name: sum(scores) / len(scores)
            for name, scores in metric_scores.items()
        }

        self.pass_rates = {
            name: sum(
                1 for r in self.sample_results
                if any(
                    (k if isinstance(k, str) else k.value) == name
                    and v.passed
                    for k, v in r.metric_results.items()
                )
            ) / len(self.sample_results)
            for name in metric_scores
        }

        self.total_samples = len(self.sample_results)
        overall_scores = [r.overall_score for r in self.sample_results]
        self.passed_samples = sum(1 for s in overall_scores if s >= 0.7)
