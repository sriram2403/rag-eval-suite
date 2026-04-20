"""
Abstract base class for all RAG evaluation metrics.
"""
from abc import ABC, abstractmethod
from typing import List
from core.models import RAGSample, MetricResult, MetricName


class BaseMetric(ABC):
    """Abstract base for all evaluation metrics."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    @property
    @abstractmethod
    def name(self) -> MetricName:
        pass

    @abstractmethod
    def compute(self, sample: RAGSample) -> MetricResult:
        pass

    def compute_batch(self, samples: List[RAGSample]) -> List[MetricResult]:
        return [self.compute(s) for s in samples]
