#!/usr/bin/env python3
"""
Example: Programmatic usage of the RAG Evaluation Framework.

Shows how to:
1. Evaluate pre-generated answers
2. Run a full pipeline + evaluate
3. Compare two pipelines
4. Use only specific metrics
5. Use custom thresholds
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.models import RAGSample, BenchmarkConfig, MetricName
from core.evaluator import RAGEvaluator


# ─── Example 1: Evaluate pre-generated answers ──────────────────────────────

def example_pregenerated():
    """Evaluate answers you've already generated (no pipeline needed)."""
    print("\n" + "="*60)
    print("Example 1: Evaluating Pre-Generated Answers")
    print("="*60)

    samples = [
        RAGSample(
            question="What causes tides on Earth?",
            ground_truth=(
                "Tides are primarily caused by the gravitational pull of the Moon "
                "and, to a lesser extent, the Sun on Earth's oceans. The Moon's "
                "gravity creates two tidal bulges: one facing the Moon and one on "
                "the opposite side. As Earth rotates, these bulges create high and "
                "low tides roughly twice a day."
            ),
            contexts=[
                "Tides result from the gravitational interaction between Earth and "
                "celestial bodies, mainly the Moon. The Moon exerts a stronger pull "
                "on the side of Earth facing it, creating a bulge of ocean water. A "
                "corresponding bulge forms on the opposite side due to inertia.",
                "The Sun also influences tides but with about 46% the effect of the Moon "
                "due to its much greater distance despite larger mass. When the Sun and "
                "Moon align (new/full moon), spring tides occur with maximum tidal range.",
            ],
            answer=(
                "Tides are caused by the gravitational pull of the Moon on Earth's "
                "oceans. The Moon's gravity creates two bulges of water, one on the "
                "side facing the Moon and one on the opposite side. As Earth rotates "
                "through these bulges, we experience high and low tides. The Sun also "
                "contributes to tides, though with less effect than the Moon."
            ),
        ),
        RAGSample(
            question="What is quantum entanglement?",
            ground_truth=(
                "Quantum entanglement is a phenomenon where two or more particles "
                "become correlated in such a way that the quantum state of each particle "
                "cannot be described independently. When one particle is measured, "
                "the state of its entangled partner is instantly determined, regardless "
                "of the distance between them."
            ),
            contexts=[
                "Quantum entanglement occurs when particles interact in ways such that "
                "the quantum state of each cannot be described independently of the others, "
                "even when separated by large distances. Einstein famously called this "
                "'spooky action at a distance.'",
            ],
            # Hallucinated answer — claims something not in context
            answer=(
                "Quantum entanglement is a phenomenon where particles are linked and "
                "can communicate faster than light. When you observe one particle, "
                "the other instantly knows what state to be in. This was proven "
                "experimentally by Albert Einstein in 1935 at Princeton University."
            ),
        ),
    ]

    config = BenchmarkConfig(
        name="pregenerated_eval",
        metrics=[
            MetricName.FAITHFULNESS,
            MetricName.GROUNDEDNESS,
            MetricName.ROUGE_L,
        ],
        use_llm_judge=True,
        verbose=True
    )

    evaluator = RAGEvaluator(config)
    report = evaluator.evaluate_dataset(samples, verbose=True)
    evaluator.print_summary(report)
    return report


# ─── Example 2: Run pipeline + evaluate ─────────────────────────────────────

def example_pipeline_eval():
    """Run a RAG pipeline and evaluate its output automatically."""
    print("\n" + "="*60)
    print("Example 2: Pipeline Evaluation (Naive RAG)")
    print("="*60)

    from datasets.benchmark_data import get_science_qa_dataset, get_corpus_documents
    from pipelines.rag_pipelines import NaiveRAGPipeline

    docs = get_corpus_documents()
    pipeline = NaiveRAGPipeline(documents=docs, top_k=3)
    samples = get_science_qa_dataset()[:2]  # First 2 samples

    config = BenchmarkConfig(
        name="naive_rag_pipeline",
        metrics=[
            MetricName.FAITHFULNESS,
            MetricName.ANSWER_RELEVANCE,
            MetricName.CONTEXT_RECALL,
            MetricName.GROUNDEDNESS,
        ],
        use_llm_judge=True,
        verbose=True
    )

    evaluator = RAGEvaluator(config)
    report = evaluator.evaluate_dataset(samples, pipeline=pipeline, verbose=True)
    evaluator.print_summary(report)
    return report


# ─── Example 3: Compare two pipelines ───────────────────────────────────────

def example_compare_pipelines():
    """Compare Naive vs Semantic RAG on the same dataset."""
    print("\n" + "="*60)
    print("Example 3: Pipeline Comparison")
    print("="*60)

    from datasets.benchmark_data import get_tech_qa_dataset, get_corpus_documents
    from pipelines.rag_pipelines import NaiveRAGPipeline, SemanticRAGPipeline

    docs = get_corpus_documents()
    samples = get_tech_qa_dataset()[:2]

    config = BenchmarkConfig(
        name="comparison",
        metrics=[MetricName.FAITHFULNESS, MetricName.GROUNDEDNESS, MetricName.ROUGE_L],
        use_llm_judge=True,
        verbose=False
    )

    pipelines = {
        "naive_tfidf": NaiveRAGPipeline(documents=docs, top_k=3),
        "semantic_emb": SemanticRAGPipeline(documents=docs, top_k=3),
    }

    results = {}
    for name, pipeline in pipelines.items():
        print(f"\n  Running: {name}")
        evaluator = RAGEvaluator(config)
        report = evaluator.evaluate_dataset(samples, pipeline=pipeline, verbose=False)
        results[name] = report

    # Side-by-side comparison
    print("\n" + "-"*60)
    print(f"{'Metric':<25} {'naive_tfidf':>15} {'semantic_emb':>15}")
    print("-"*60)

    all_metrics = list(results["naive_tfidf"].aggregate_scores.keys())
    for metric in all_metrics:
        scores = [results[p].aggregate_scores.get(metric, 0) for p in pipelines]
        best = max(scores)
        row = f"{metric:<25}"
        for score in scores:
            marker = " ★" if score == best else "  "
            row += f"{score:.3f}{marker:>12}"
        print(row)

    print("-"*60)
    for name, report in results.items():
        overall = sum(report.aggregate_scores.values()) / max(len(report.aggregate_scores), 1)
        print(f"  {name}: overall={overall:.3f}, duration={report.run_duration_s:.1f}s")

    return results


# ─── Example 4: Custom metric thresholds ────────────────────────────────────

def example_custom_thresholds():
    """Use strict custom thresholds for production quality gates."""
    print("\n" + "="*60)
    print("Example 4: Custom Strict Thresholds (Production Gate)")
    print("="*60)

    from datasets.benchmark_data import get_hallucination_test_dataset, get_corpus_documents
    from pipelines.rag_pipelines import NaiveRAGPipeline

    docs = get_corpus_documents()
    pipeline = NaiveRAGPipeline(documents=docs, top_k=3)
    samples = get_hallucination_test_dataset()

    # Strict thresholds for production
    config = BenchmarkConfig(
        name="production_gate",
        metrics=[MetricName.FAITHFULNESS, MetricName.GROUNDEDNESS],
        thresholds={
            "faithfulness": 0.85,    # Very strict — we hate hallucinations
            "groundedness": 0.80,    # Strict — answers must stay in context
        },
        use_llm_judge=True,
        verbose=True
    )

    evaluator = RAGEvaluator(config)
    report = evaluator.evaluate_dataset(samples, pipeline=pipeline, verbose=True)
    evaluator.print_summary(report)

    # Quality gate decision
    print("\n🚦 Production Quality Gate:")
    all_passed = all(
        report.aggregate_scores.get(m, 0) >= config.thresholds.get(m, 0.7)
        for m in [mn.value for mn in config.metrics]
    )
    if all_passed:
        print("  ✅ PASSED — Pipeline meets production standards")
    else:
        print("  ❌ FAILED — Pipeline does not meet production standards")
        failing = [
            m for m in [mn.value for mn in config.metrics]
            if report.aggregate_scores.get(m, 0) < config.thresholds.get(m, 0.7)
        ]
        print(f"  Failing metrics: {', '.join(failing)}")

    return report, all_passed


# ─── Example 5: Single sample deep dive ─────────────────────────────────────

def example_single_sample_analysis():
    """Deep analysis of a single sample."""
    print("\n" + "="*60)
    print("Example 5: Single Sample Deep Analysis")
    print("="*60)

    sample = RAGSample(
        question="How does HTTPS encryption protect data?",
        ground_truth=(
            "HTTPS uses TLS to encrypt web traffic. During the TLS handshake, "
            "the server presents a digital certificate. The client verifies the "
            "certificate, and both parties establish an encrypted session. Data "
            "transmitted is protected from interception and tampering."
        ),
        contexts=[
            "HTTPS uses TLS (Transport Layer Security) encryption. The TLS handshake "
            "involves: server sending certificate, client verifying with CA, both parties "
            "deriving session keys for symmetric encryption of all data.",
            "TLS 1.3 provides forward secrecy using ephemeral key pairs. Even if keys "
            "are later compromised, past sessions remain protected."
        ],
        answer=(
            "HTTPS protects data using TLS encryption. When you connect to an HTTPS "
            "site, the server presents a certificate to prove its identity. After "
            "verification, both sides establish shared encryption keys. All data "
            "transferred is encrypted, preventing eavesdropping and tampering."
        )
    )

    from core.models import BenchmarkConfig, MetricName
    config = BenchmarkConfig(
        name="deep_analysis",
        metrics=[
            MetricName.FAITHFULNESS,
            MetricName.ANSWER_RELEVANCE,
            MetricName.CONTEXT_RECALL,
            MetricName.GROUNDEDNESS,
            MetricName.ROUGE_L,
        ],
        use_llm_judge=True,
        verbose=False
    )

    evaluator = RAGEvaluator(config)
    result = evaluator.evaluate_sample(sample)

    print(f"\n  Question: {sample.question}")
    print(f"  Overall Score: {result.overall_score:.3f}")
    print()

    for metric_name, metric_result in result.metric_results.items():
        status = "✓" if metric_result.passed else "✗"
        print(f"  {status} {metric_name:<25}: {metric_result.score:.3f}")
        print(f"      {metric_result.explanation[:80]}")
        print()

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Eval Framework Examples")
    parser.add_argument("--example", type=int, default=5,
                        choices=[1, 2, 3, 4, 5],
                        help="Which example to run (1-5)")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    args = parser.parse_args()

    examples = {
        1: ("Pre-generated Answers", example_pregenerated),
        2: ("Pipeline Evaluation", example_pipeline_eval),
        3: ("Compare Pipelines", example_compare_pipelines),
        4: ("Custom Thresholds", example_custom_thresholds),
        5: ("Single Sample Analysis", example_single_sample_analysis),
    }

    if args.all:
        for num, (name, fn) in examples.items():
            fn()
    else:
        name, fn = examples[args.example]
        fn()
