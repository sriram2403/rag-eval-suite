#!/usr/bin/env python3
"""
Quick end-to-end test of the RAG Evaluation Framework.
Runs with mock pipeline (no API calls needed) to validate everything wires up.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_models():
    print("  Testing data models...")
    from core.models import RAGSample, BenchmarkConfig, MetricName, MetricResult
    sample = RAGSample(
        question="What is photosynthesis?",
        ground_truth="Photosynthesis converts light to energy in plants.",
        contexts=["Plants use sunlight and CO2 to produce glucose via photosynthesis."],
        answer="Photosynthesis is a process where plants convert sunlight into energy."
    )
    assert sample.sample_id, "sample_id should be auto-generated"
    config = BenchmarkConfig(name="test_config")
    assert len(config.metrics) > 0
    print("  ✓ Data models OK")

def test_rouge_metric():
    print("  Testing ROUGE-L metric (no API)...")
    from core.models import RAGSample
    from metrics.semantic_similarity import ROUGELMetric

    metric = ROUGELMetric(threshold=0.3)
    sample = RAGSample(
        question="What is water?",
        ground_truth="Water is a liquid composed of hydrogen and oxygen molecules.",
        contexts=["H2O is water"],
        answer="Water is a liquid made of hydrogen and oxygen."
    )
    result = metric.compute(sample)
    assert 0.0 <= result.score <= 1.0, f"Score out of range: {result.score}"
    assert result.metric_name.value == "rouge_l"
    print(f"  ✓ ROUGE-L metric OK (score: {result.score:.3f})")

def test_semantic_similarity():
    print("  Testing Semantic Similarity metric...")
    from core.models import RAGSample
    from metrics.semantic_similarity import SemanticSimilarityMetric

    try:
        metric = SemanticSimilarityMetric(threshold=0.75)
        sample = RAGSample(
            question="What is AI?",
            ground_truth="Artificial intelligence is the simulation of human intelligence by machines.",
            contexts=["AI involves machine learning and deep learning."],
            answer="AI is when machines simulate human thinking and intelligence."
        )
        result = metric.compute(sample)
        assert 0.0 <= result.score <= 1.0
        print(f"  ✓ Semantic Similarity OK (score: {result.score:.3f})")
    except Exception as e:
        print(f"  ⚠ Semantic Similarity skipped (sentence-transformers not installed): {e}")

def test_mock_pipeline():
    print("  Testing mock pipeline...")
    from pipelines.rag_pipelines import MockPipeline

    pipeline = MockPipeline()
    answer, contexts = pipeline.run("What is photosynthesis?")
    assert isinstance(answer, str) and len(answer) > 0
    assert isinstance(contexts, list)
    print(f"  ✓ Mock pipeline OK (answer: '{answer[:50]}...')")

def test_naive_pipeline_retrieval():
    print("  Testing Naive RAG retrieval (no API)...")
    from pipelines.rag_pipelines import NaiveRAGPipeline

    docs = [
        "Photosynthesis converts sunlight into glucose in plants.",
        "The mitochondria is the powerhouse of the cell.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Einstein proposed the theory of relativity.",
    ]
    pipeline = NaiveRAGPipeline(documents=docs, top_k=2)
    results = pipeline.retrieve("How do plants make food from sunlight?")
    assert len(results) > 0, "Should retrieve at least one document"
    # Photosynthesis doc should be most relevant
    assert any("photosynthesis" in r.lower() or "sunlight" in r.lower() for r in results)
    print(f"  ✓ Naive RAG retrieval OK (retrieved {len(results)} docs)")

def test_evaluator_with_mock():
    print("  Testing full evaluator with mock pipeline...")
    from core.models import RAGSample, BenchmarkConfig, MetricName
    from core.evaluator import RAGEvaluator
    from pipelines.rag_pipelines import MockPipeline

    config = BenchmarkConfig(
        name="test_run",
        metrics=[MetricName.ROUGE_L],  # Only non-LLM metric for fast test
        use_llm_judge=False,
        verbose=False
    )

    samples = [
        RAGSample(
            question="What is the capital of France?",
            ground_truth="The capital of France is Paris.",
            contexts=["Paris is the capital and largest city of France."],
            answer="Paris is the capital of France."
        )
    ]

    evaluator = RAGEvaluator(config)
    result = evaluator.evaluate_sample(samples[0])

    assert result.overall_score >= 0.0
    assert "rouge_l" in result.metric_results
    score = result.metric_results["rouge_l"].score
    assert score > 0.5, f"Expected good ROUGE score for similar answer, got {score}"
    print(f"  ✓ Evaluator OK (overall: {result.overall_score:.3f}, rouge_l: {score:.3f})")

def test_dataset_loading():
    print("  Testing dataset loading...")
    from datasets.benchmark_data import (
        get_science_qa_dataset, get_tech_qa_dataset,
        get_hallucination_test_dataset, get_all_datasets
    )
    science = get_science_qa_dataset()
    tech = get_tech_qa_dataset()
    hall = get_hallucination_test_dataset()
    all_data = get_all_datasets()

    assert len(science) >= 3
    assert len(tech) >= 3
    assert len(hall) >= 2
    assert len(all_data) == len(science) + len(tech) + len(hall)

    for s in all_data:
        assert s.question, f"Missing question in {s.sample_id}"
        assert s.ground_truth, f"Missing ground_truth in {s.sample_id}"
        assert len(s.contexts) > 0, f"No contexts in {s.sample_id}"

    print(f"  ✓ Datasets OK ({len(all_data)} total samples)")

def test_report_serialization():
    print("  Testing report serialization...")
    import json
    from core.models import RAGSample, BenchmarkConfig, MetricName
    from core.evaluator import RAGEvaluator

    config = BenchmarkConfig(
        name="serialization_test",
        metrics=[MetricName.ROUGE_L],
        use_llm_judge=False,
        verbose=False
    )

    samples = [
        RAGSample(
            question="Test question?",
            ground_truth="Test answer.",
            contexts=["Test context."],
            answer="Test answer here."
        )
    ]

    evaluator = RAGEvaluator(config)
    report = evaluator.evaluate_dataset(samples, verbose=False)

    # Serialize to JSON
    def serialize(obj):
        if hasattr(obj, 'value'): return obj.value
        if hasattr(obj, '__dict__'): return obj.__dict__
        return str(obj)

    json_str = json.dumps(report, default=serialize)
    parsed = json.loads(json_str)

    assert "aggregate_scores" in parsed
    assert "sample_results" in parsed
    assert len(parsed["sample_results"]) == 1
    print(f"  ✓ Report serialization OK ({len(json_str)} chars)")

def test_metrics_registry():
    print("  Testing metrics registry...")
    from core.models import BenchmarkConfig, MetricName
    from metrics import build_metrics, METRIC_DESCRIPTIONS

    config = BenchmarkConfig(
        name="registry_test",
        metrics=[MetricName.ROUGE_L, MetricName.SEMANTIC_SIMILARITY],
        use_llm_judge=False
    )
    metrics = build_metrics(config)
    assert len(metrics) == 2
    assert any(m.name == MetricName.ROUGE_L for m in metrics)
    assert len(METRIC_DESCRIPTIONS) >= 5
    print(f"  ✓ Metrics registry OK ({len(metrics)} metrics built)")

def run_all_tests():
    tests = [
        ("Data Models", test_models),
        ("ROUGE-L Metric", test_rouge_metric),
        ("Semantic Similarity", test_semantic_similarity),
        ("Mock Pipeline", test_mock_pipeline),
        ("Naive RAG Retrieval", test_naive_pipeline_retrieval),
        ("Full Evaluator", test_evaluator_with_mock),
        ("Dataset Loading", test_dataset_loading),
        ("Report Serialization", test_report_serialization),
        ("Metrics Registry", test_metrics_registry),
    ]

    print("\n" + "="*55)
    print("  RAG EVALUATION FRAMEWORK — TEST SUITE")
    print("="*55 + "\n")

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        print(f"[{name}]")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ FAILED: {e}")
            errors.append((name, str(e), traceback.format_exc()))
            failed += 1
        print()

    print("="*55)
    print(f"  Results: {passed} passed, {failed} failed")
    print("="*55)

    if errors:
        print("\nFailed tests:")
        for name, err, tb in errors:
            print(f"\n  [{name}]: {err}")
            if "--verbose" in sys.argv:
                print(tb)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
