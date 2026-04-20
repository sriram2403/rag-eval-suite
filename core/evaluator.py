"""
RAG Evaluator — orchestrates evaluation of RAG pipelines.

Usage:
    evaluator = RAGEvaluator(config)
    report = evaluator.evaluate_dataset(samples, pipeline)
"""
import time
import json
import os
from typing import List, Optional
from datetime import datetime

from core.models import (
    RAGSample, BenchmarkConfig, BenchmarkReport,
    SampleEvalResult, MetricName
)
from core.base_metric import BaseMetric
from metrics import build_metrics
from pipelines.rag_pipelines import BasePipeline


class RAGEvaluator:
    """
    Central evaluation engine for RAG pipelines.

    Supports:
    - Evaluating pre-generated answers (metrics only)
    - Running a pipeline and evaluating (retrieve + generate + evaluate)
    - Batch evaluation with progress tracking
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics: List[BaseMetric] = build_metrics(config)

    def evaluate_sample(self, sample: RAGSample) -> SampleEvalResult:
        """Evaluate a single sample (answer must already be set)."""
        result = SampleEvalResult(sample=sample)
        start = time.time()

        for metric in self.metrics:
            try:
                metric_result = metric.compute(sample)
                result.metric_results[metric.name.value] = metric_result
            except Exception as e:
                result.errors.append(f"{metric.name.value}: {str(e)}")

        result.latency_ms = (time.time() - start) * 1000
        result.compute_overall()
        return result

    def evaluate_with_pipeline(
        self, sample: RAGSample, pipeline: BasePipeline
    ) -> SampleEvalResult:
        """Run the pipeline and then evaluate the output."""
        start = time.time()
        try:
            answer, contexts = pipeline.run(sample.question)
            # Fill in pipeline outputs
            sample = RAGSample(
                sample_id=sample.sample_id,
                question=sample.question,
                ground_truth=sample.ground_truth,
                contexts=contexts if contexts else sample.contexts,
                answer=answer,
                metadata=sample.metadata
            )
        except Exception as e:
            sample = RAGSample(
                sample_id=sample.sample_id,
                question=sample.question,
                ground_truth=sample.ground_truth,
                contexts=sample.contexts,
                answer="",
                metadata=sample.metadata
            )
            result = SampleEvalResult(sample=sample, errors=[f"Pipeline error: {e}"])
            return result

        pipeline_latency_ms = (time.time() - start) * 1000
        result = self.evaluate_sample(sample)
        result.latency_ms += pipeline_latency_ms
        return result

    def evaluate_dataset(
        self,
        samples: List[RAGSample],
        pipeline: Optional[BasePipeline] = None,
        verbose: bool = True
    ) -> BenchmarkReport:
        """
        Evaluate a full dataset.

        If pipeline is provided, it will generate answers.
        Otherwise, samples must already have answers set.
        """
        config = self.config
        max_samples = config.max_samples
        if max_samples:
            samples = samples[:max_samples]

        report = BenchmarkReport(
            config=config,
            pipeline_name=pipeline.name if pipeline else "pre-generated",
            timestamp=datetime.now().isoformat()
        )

        start_total = time.time()

        for i, sample in enumerate(samples):
            if verbose:
                print(f"  [{i+1}/{len(samples)}] Evaluating: {sample.question[:60]}...")

            if pipeline:
                result = self.evaluate_with_pipeline(sample, pipeline)
            else:
                if not sample.answer:
                    result = SampleEvalResult(
                        sample=sample,
                        errors=["No answer provided and no pipeline specified"]
                    )
                else:
                    result = self.evaluate_sample(sample)

            report.sample_results.append(result)

            if verbose and result.metric_results:
                scores_str = "  ".join(
                    f"{k}: {v.score:.2f}" for k, v in result.metric_results.items()
                )
                status = "✓" if result.overall_score >= 0.7 else "✗"
                print(f"    {status} Overall: {result.overall_score:.3f} | {scores_str}")

        report.run_duration_s = time.time() - start_total
        report.compute_aggregates()
        return report

    def save_report(self, report: BenchmarkReport, output_path: str):
        """Save the benchmark report to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        def serialize(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            if hasattr(obj, 'value'):
                return obj.value
            return str(obj)

        with open(output_path, 'w') as f:
            json.dump(report, f, default=serialize, indent=2)
        print(f"\n✓ Report saved to: {output_path}")

    def print_summary(self, report: BenchmarkReport):
        """Print a rich summary of the benchmark report."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich import box

            console = Console()

            console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            console.print(f"[bold]RAG Benchmark: {report.config.name}[/bold]")
            console.print(f"Pipeline: [yellow]{report.pipeline_name}[/yellow]")
            console.print(f"Samples: {report.total_samples} | Duration: {report.run_duration_s:.1f}s")
            console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

            # Metric scores table
            table = Table(title="Metric Scores", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Avg Score", justify="center")
            table.add_column("Pass Rate", justify="center")
            table.add_column("Threshold", justify="center")
            table.add_column("Status", justify="center")

            for metric_name, score in report.aggregate_scores.items():
                pass_rate = report.pass_rates.get(metric_name, 0)
                threshold = report.config.thresholds.get(metric_name, 0.7)
                status = "✅ PASS" if score >= threshold else "❌ FAIL"
                color = "green" if score >= threshold else "red"

                table.add_row(
                    metric_name.replace("_", " ").title(),
                    f"[{color}]{score:.3f}[/{color}]",
                    f"{pass_rate:.1%}",
                    f"{threshold:.2f}",
                    status
                )

            console.print(table)

            # Overall summary
            overall_avg = sum(report.aggregate_scores.values()) / max(len(report.aggregate_scores), 1)
            color = "green" if overall_avg >= 0.7 else "yellow" if overall_avg >= 0.5 else "red"
            console.print(f"\n[bold]Overall Score: [{color}]{overall_avg:.3f}[/{color}][/bold]")
            console.print(f"Samples Passed: {report.passed_samples}/{report.total_samples} ({report.passed_samples/max(report.total_samples,1):.1%})")

        except ImportError:
            # Plain text fallback
            print(f"\n{'='*60}")
            print(f"RAG Benchmark: {report.config.name}")
            print(f"Pipeline: {report.pipeline_name}")
            print(f"{'='*60}")
            for metric_name, score in report.aggregate_scores.items():
                threshold = report.config.thresholds.get(metric_name, 0.7)
                status = "PASS" if score >= threshold else "FAIL"
                print(f"  {metric_name:25s}: {score:.3f} [{status}]")
            overall = sum(report.aggregate_scores.values()) / max(len(report.aggregate_scores), 1)
            print(f"\nOverall: {overall:.3f}")
