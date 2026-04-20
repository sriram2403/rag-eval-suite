"""
FastAPI server for the RAG Evaluation Dashboard.
Provides REST API endpoints for running evaluations and viewing results.
"""
import sys
import os
import json
import time
import shutil
from fastapi import UploadFile, File
from typing import Optional, List
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.models import BenchmarkConfig, MetricName
from core.evaluator import RAGEvaluator

app = FastAPI(title="RAG Evaluation Framework", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for benchmark jobs and results
jobs = {}
reports = {}


class BenchmarkRequest(BaseModel):
    pipeline: str = "naive"
    dataset: str = "science"
    metrics: List[str] = ["faithfulness", "answer_relevance", "context_recall", "groundedness"]
    use_llm: bool = True
    max_samples: Optional[int] = 3


class CustomSampleRequest(BaseModel):
    question: str
    ground_truth: str
    contexts: List[str]
    answer: str = ""
    pipeline: str = "naive"


def get_dataset(name: str):
    from datasets.benchmark_data import (
        get_science_qa_dataset, get_tech_qa_dataset,
        get_hallucination_test_dataset, get_all_datasets
    )
    dataset_map = {
        "science": get_science_qa_dataset,
        "tech": get_tech_qa_dataset,
        "hallucination": get_hallucination_test_dataset,
        "all": get_all_datasets,
    }
    if name not in dataset_map:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {name}")
    return dataset_map[name]()


def get_pipeline(name: str):
    from pipelines.rag_pipelines import NaiveRAGPipeline, SemanticRAGPipeline, MockPipeline
    from datasets.benchmark_data import get_corpus_documents
    docs = get_corpus_documents()

    if name == "naive":
        return NaiveRAGPipeline(documents=docs, top_k=3)
    elif name == "semantic":
        return SemanticRAGPipeline(documents=docs, top_k=3)
    elif name == "mock":
        return MockPipeline()
    raise HTTPException(status_code=400, detail=f"Unknown pipeline: {name}")


def run_benchmark_sync(job_id: str, request: BenchmarkRequest):
    """Run benchmark synchronously (called in thread pool)."""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = datetime.now().isoformat()

        metric_list = [MetricName(m) for m in request.metrics if m in [mn.value for mn in MetricName]]

        config = BenchmarkConfig(
            name=f"{request.pipeline}_{request.dataset}",
            metrics=metric_list,
            use_llm_judge=request.use_llm,
            max_samples=request.max_samples,
            verbose=False
        )

        samples = get_dataset(request.dataset)
        pipeline = get_pipeline(request.pipeline)
        evaluator = RAGEvaluator(config)

        report = evaluator.evaluate_dataset(samples, pipeline=pipeline, verbose=False)

        # Serialize report
        def serialize(obj):
            if hasattr(obj, 'value'):
                return obj.value
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)

        report_data = json.loads(json.dumps(report, default=serialize))
        reports[job_id] = report_data
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        import traceback
        jobs[job_id]["traceback"] = traceback.format_exc()


_base = os.path.dirname(os.path.abspath(__file__))
_dist = os.path.join(_base, "dashboard", "dist")
_assets = os.path.join(_dist, "assets")

if os.path.isdir(_assets):
    app.mount("/assets", StaticFiles(directory=_assets), name="static_assets")


@app.get("/")
async def serve_dashboard():
    """Serve the React dashboard (built) or fallback HTML."""
    for path in [
        os.path.join(_dist, "index.html"),
        os.path.join(_base, "dashboard", "index.html"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                return HTMLResponse(f.read())
    return HTMLResponse("<h1>Dashboard not found — run: cd dashboard && npm run build</h1>")


@app.post("/api/benchmark")
async def start_benchmark(request: BenchmarkRequest):
    """Start a benchmark evaluation job."""
    job_id = f"job_{int(time.time() * 1000)}"
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "request": request.model_dump(),
        "created_at": datetime.now().isoformat()
    }

    import threading
    thread = threading.Thread(
        target=run_benchmark_sync,
        args=(job_id, request),
        daemon=True
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a benchmark job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get the results of a completed benchmark job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["status"] != "completed":
        return {"status": jobs[job_id]["status"], "message": "Job not completed yet"}
    if job_id not in reports:
        raise HTTPException(status_code=404, detail="Results not found")
    return reports[job_id]


@app.post("/api/evaluate-sample")
async def evaluate_sample(request: CustomSampleRequest):
    from core.models import RAGSample, BenchmarkConfig, MetricName
    from core.evaluator import RAGEvaluator

    sample = RAGSample(
        question=request.question,
        ground_truth=request.ground_truth,
        contexts=request.contexts,
        answer=request.answer,
    )

    if not request.answer:
        # Add the user's pasted contexts INTO the pipeline's document pool
        from datasets.benchmark_data import get_corpus_documents
        from pipelines.rag_pipelines import NaiveRAGPipeline

        all_docs = request.contexts + get_corpus_documents()
        pipeline = NaiveRAGPipeline(documents=all_docs, top_k=3)

        retrieved_contexts = pipeline.retrieve(request.question)
        answer = pipeline.generate(request.question, retrieved_contexts)

        sample.answer = answer
        sample.contexts = retrieved_contexts  # use what was actually retrieved

    config = BenchmarkConfig(
        name="single_sample",
        metrics=[
            MetricName.FAITHFULNESS,
            MetricName.ANSWER_RELEVANCE,
            MetricName.GROUNDEDNESS,
            MetricName.CONTEXT_RECALL,
        ],
        use_llm_judge=True,
        verbose=False
    )

    evaluator = RAGEvaluator(config)
    result = evaluator.evaluate_sample(sample)

    def serialize(obj):
        if hasattr(obj, 'value'):
            return obj.value
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)

    return json.loads(json.dumps(result, default=serialize))


@app.get("/api/metrics")
async def list_metrics():
    """List all available metrics."""
    from metrics import METRIC_DESCRIPTIONS
    return {
        m.value: desc for m, desc in METRIC_DESCRIPTIONS.items()
    }


@app.get("/api/datasets")
async def list_datasets():
    """List all available datasets."""
    from datasets.benchmark_data import (
        get_science_qa_dataset, get_tech_qa_dataset,
        get_hallucination_test_dataset, get_all_datasets
    )
    datasets = {
        "science": get_science_qa_dataset,
        "tech": get_tech_qa_dataset,
        "hallucination": get_hallucination_test_dataset,
        "all": get_all_datasets,
    }
    return {
        name: {
            "count": len(fn()),
            "domains": list(set(s.metadata.get("domain", "?") for s in fn()))
        }
        for name, fn in datasets.items()
    }


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    return list(jobs.values())


UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document into Supabase."""
    allowed_types = ['.pdf', '.docx', '.doc', '.txt', '.md', '.yaml', '.yml', '.xml', '.json', '.html', '.htm', '.rtf']
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_types}"
        )

    # Save file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    try:
        from document_processor import upload_document
        result = upload_document(file_path, file.filename)
        return {
            "success": True,
            "filename": file.filename,
            "chunks": result['total_chunks'],
            "characters": result['total_characters']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def list_documents():
    """List all uploaded documents."""
    from document_processor import list_documents
    docs = list_documents()
    return {"documents": docs}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document from Supabase."""
    from document_processor import delete_document
    delete_document(filename)
    return {"success": True, "deleted": filename}


@app.post("/api/ask")
async def ask_document(request: dict):
    """Ask a question against uploaded documents."""
    question = request.get("question", "")
    filename = request.get("filename")  # optional: filter to specific doc
    ground_truth = request.get("ground_truth", "")

    if not question:
        raise HTTPException(status_code=400, detail="Question required")

    from pipelines.supabase_pipeline import SupabaseRAGPipeline
    from core.models import RAGSample, BenchmarkConfig, MetricName
    from core.evaluator import RAGEvaluator

    pipeline = SupabaseRAGPipeline(top_k=4, filename=filename)
    answer, contexts = pipeline.run(question)

    result_data = {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }

    # Evaluate if ground truth provided
    if ground_truth:
        sample = RAGSample(
            question=question,
            ground_truth=ground_truth,
            contexts=contexts,
            answer=answer
        )
        config = BenchmarkConfig(
            name="doc_eval",
            metrics=[
                MetricName.FAITHFULNESS,
                MetricName.ANSWER_RELEVANCE,
                MetricName.GROUNDEDNESS,
                MetricName.CONTEXT_RECALL,
            ],
            use_llm_judge=True,
            verbose=False
        )
        evaluator = RAGEvaluator(config)
        eval_result = evaluator.evaluate_sample(sample)

        def serialize(obj):
            if hasattr(obj, 'value'): return obj.value
            if hasattr(obj, '__dict__'): return obj.__dict__
            return str(obj)

        import json
        result_data["evaluation"] = json.loads(
            json.dumps(eval_result, default=serialize)
        )

    return result_data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
