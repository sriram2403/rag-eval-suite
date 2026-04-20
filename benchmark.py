#!/usr/bin/env python3
"""
RAG Evaluation Framework — CLI

Usage:
  python benchmark.py run --pipeline naive --dataset science
  python benchmark.py run --pipeline semantic --dataset all --metrics faithfulness,groundedness
  python benchmark.py compare --pipelines naive,semantic --dataset tech
  python benchmark.py list-metrics
  python benchmark.py list-datasets
"""
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
from core.models import BenchmarkConfig, MetricName
from core.evaluator import RAGEvaluator
from datasets.benchmark_data import (
    get_science_qa_dataset, get_tech_qa_dataset,
    get_hallucination_test_dataset, get_all_datasets,
    get_corpus_documents
)

DATASET_MAP = {
    "science": get_science_qa_dataset,
    "tech": get_tech_qa_dataset,
    "hallucination": get_hallucination_test_dataset,
    "all": get_all_datasets,
}

METRIC_MAP = {m.value: m for m in MetricName}


def get_pipeline(pipeline_name: str):
    """Factory function for pipelines."""
    from pipelines.rag_pipelines import NaiveRAGPipeline, SemanticRAGPipeline, MockPipeline
    docs = get_corpus_documents()

    if pipeline_name == "naive":
        return NaiveRAGPipeline(documents=docs, top_k=3)
    elif pipeline_name == "semantic":
        return SemanticRAGPipeline(documents=docs, top_k=3)
    elif pipeline_name == "mock":
        return MockPipeline()
    else:
        click.echo(f"Unknown pipeline: {pipeline_name}. Available: naive, semantic, mock")
        sys.exit(1)


@click.group()
def cli():
    """🔍 RAG Evaluation Framework — benchmark your RAG pipelines."""
    pass


@cli.command()
@click.option("--pipeline", "-p", default="naive",
              help="Pipeline to evaluate: naive, semantic, mock")
@click.option("--dataset", "-d", default="all",
              help="Dataset: science, tech, hallucination, all")
@click.option("--metrics", "-m", default="faithfulness,answer_relevance,context_recall,groundedness",
              help="Comma-separated list of metrics")
@click.option("--output", "-o", default="reports/report.json",
              help="Output path for JSON report")
@click.option("--no-llm", is_flag=True, default=False,
              help="Use only non-LLM metrics (faster, no API cost)")
@click.option("--max-samples", "-n", default=None, type=int,
              help="Maximum number of samples to evaluate")
@click.option("--verbose/--quiet", default=True)
def run(pipeline, dataset, metrics, output, no_llm, max_samples, verbose):
    """Run a benchmark evaluation."""
    click.echo(f"\n🚀 RAG Evaluation Framework")
    click.echo(f"   Pipeline: {pipeline}")
    click.echo(f"   Dataset:  {dataset}")
    click.echo(f"   Metrics:  {metrics}\n")

    # Build config
    metric_list = [
        METRIC_MAP[m.strip()] for m in metrics.split(",")
        if m.strip() in METRIC_MAP
    ]

    config = BenchmarkConfig(
        name=f"{pipeline}_{dataset}_benchmark",
        metrics=metric_list,
        use_llm_judge=not no_llm,
        max_samples=max_samples,
        verbose=verbose
    )

    # Load dataset
    if dataset not in DATASET_MAP:
        click.echo(f"Unknown dataset: {dataset}. Options: {list(DATASET_MAP.keys())}")
        sys.exit(1)

    samples = DATASET_MAP[dataset]()
    click.echo(f"📂 Loaded {len(samples)} samples from '{dataset}' dataset")

    # Load pipeline
    rag_pipeline = get_pipeline(pipeline)
    click.echo(f"🔧 Initialized pipeline: {rag_pipeline.name}\n")

    # Run evaluation
    evaluator = RAGEvaluator(config)
    click.echo("⚡ Running evaluation...\n")

    report = evaluator.evaluate_dataset(samples, pipeline=rag_pipeline, verbose=verbose)

    # Print summary
    evaluator.print_summary(report)

    # Save report
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    evaluator.save_report(report, output)

    return report


@cli.command()
@click.option("--pipelines", "-p", default="naive,semantic",
              help="Comma-separated pipelines to compare")
@click.option("--dataset", "-d", default="all")
@click.option("--metrics", "-m", default="faithfulness,answer_relevance,groundedness")
@click.option("--max-samples", "-n", default=3, type=int)
def compare(pipelines, dataset, metrics, max_samples):
    """Compare multiple RAG pipelines side by side."""
    pipeline_names = [p.strip() for p in pipelines.split(",")]

    click.echo(f"\n🔬 Pipeline Comparison")
    click.echo(f"   Pipelines: {', '.join(pipeline_names)}")
    click.echo(f"   Dataset:   {dataset}\n")

    metric_list = [
        METRIC_MAP[m.strip()] for m in metrics.split(",")
        if m.strip() in METRIC_MAP
    ]

    samples = DATASET_MAP.get(dataset, get_all_datasets)()[:max_samples]
    results = {}

    for pipeline_name in pipeline_names:
        click.echo(f"\n📊 Evaluating: {pipeline_name}")
        config = BenchmarkConfig(
            name=f"compare_{pipeline_name}",
            metrics=metric_list,
            max_samples=max_samples
        )
        pipeline = get_pipeline(pipeline_name)
        evaluator = RAGEvaluator(config)
        report = evaluator.evaluate_dataset(samples, pipeline=pipeline, verbose=False)
        results[pipeline_name] = report

    # Print comparison table
    click.echo("\n" + "="*70)
    click.echo("PIPELINE COMPARISON RESULTS")
    click.echo("="*70)

    all_metrics = list(results[pipeline_names[0]].aggregate_scores.keys())
    header = f"{'Metric':<25}" + "".join(f"{p:>15}" for p in pipeline_names)
    click.echo(header)
    click.echo("-"*70)

    for metric in all_metrics:
        row = f"{metric:<25}"
        scores = [results[p].aggregate_scores.get(metric, 0) for p in pipeline_names]
        best_score = max(scores)
        for i, (p, score) in enumerate(zip(pipeline_names, scores)):
            marker = " ★" if score == best_score else "  "
            row += f"{score:.3f}{marker:>12}"
        click.echo(row)

    click.echo("-"*70)
    overall_row = f"{'OVERALL':<25}"
    for p in pipeline_names:
        scores = list(results[p].aggregate_scores.values())
        overall = sum(scores) / max(len(scores), 1)
        overall_row += f"{overall:.3f}{'':>12}"
    click.echo(overall_row)

    # Save comparison report
    os.makedirs("reports", exist_ok=True)
    comparison_data = {
        "pipelines": pipeline_names,
        "dataset": dataset,
        "results": {
            p: {
                "aggregate_scores": r.aggregate_scores,
                "pass_rates": r.pass_rates,
                "total_samples": r.total_samples,
            }
            for p, r in results.items()
        }
    }
    with open("reports/comparison.json", "w") as f:
        json.dump(comparison_data, f, indent=2)
    click.echo(f"\n✓ Comparison saved to reports/comparison.json")


@cli.command("list-metrics")
def list_metrics():
    """List all available evaluation metrics."""
    from metrics import METRIC_DESCRIPTIONS
    click.echo("\n📏 Available Metrics:\n")
    for metric, description in METRIC_DESCRIPTIONS.items():
        click.echo(f"  {metric.value:<25} {description}")


@cli.command("list-datasets")
def list_datasets():
    """List available built-in datasets."""
    click.echo("\n📂 Available Datasets:\n")
    for name, fn in DATASET_MAP.items():
        samples = fn()
        domains = list(set(s.metadata.get("domain", "?") for s in samples))
        click.echo(f"  {name:<20} {len(samples)} samples | domains: {', '.join(domains)}")


if __name__ == "__main__":
    cli()
