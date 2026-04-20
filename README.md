# RAG Evaluation Framework

A production-grade tool for benchmarking and testing Retrieval-Augmented Generation (RAG) pipelines. Comes with a React dashboard, document Q&A with Supabase vector search, a CLI, and a REST API.

---

## Features

- **Benchmark built-in pipelines** against curated QA datasets
- **Upload your own documents** (PDF, DOCX, TXT, MD, YAML, XML, JSON, HTML, RTF) and ask questions
- **Evaluate any answer** with 7 metrics — using Groq (Llama 3) as the LLM judge
- **React dashboard** with real-time job polling and per-sample score breakdowns
- **REST API** for programmatic access

---

## Architecture

```text
rag_eval/
├── core/
│   ├── models.py            # Data models (RAGSample, BenchmarkConfig, BenchmarkReport)
│   ├── base_metric.py       # Abstract base class for all metrics
│   └── evaluator.py         # Central RAGEvaluator orchestration engine
│
├── metrics/
│   ├── __init__.py          # Metrics registry + build_metrics() factory
│   ├── fact_accuracy.py      # Fact accuracy (anti-hallucination)
│   ├── answer_quality.py     # Answers the question (reverse question generation)
│   ├── context_coverage.py   # Info coverage + source quality
│   ├── document_grounding.py # Stays in document (LLM-as-judge)
│   └── semantic_similarity.py # Meaning match + word overlap
│
├── pipelines/
│   ├── rag_pipelines.py     # Naive (TF-IDF), Semantic (embeddings), Mock
│   └── supabase_pipeline.py # Production pipeline: hybrid retrieval from Supabase
│
├── datasets/
│   └── benchmark_data.py    # Built-in QA datasets (science, tech, hallucination)
│
├── dashboard/               # React + Vite frontend
│   ├── src/
│   └── dist/                # Built output (generated — not committed)
│
├── tests/
│   └── test_framework.py    # Test suite (no LLM required)
│
├── server.py                # FastAPI REST server
├── benchmark.py             # CLI tool
├── document_processor.py    # Extract → chunk → embed → Supabase
├── groq_client.py           # Groq (Llama 3) LLM wrapper
├── examples.py              # Programmatic usage examples
└── requirements.txt
```

---

## Metrics

| Metric | Display Name | Description | LLM Required |
| ------ | ------------ | ----------- | ------------ |
| `faithfulness` | **Fact Accuracy** | Fraction of answer claims supported by retrieved context | Optional |
| `answer_relevance` | **Answers the Question** | Does the answer address what was actually asked? | Optional |
| `context_recall` | **Info Coverage** | How much of the ground truth appears in retrieved context? | Optional |
| `context_precision` | **Source Quality** | What fraction of retrieved chunks are truly relevant? | Optional |
| `groundedness` | **Stays in Document** | Holistic LLM-judge check that answer doesn't go beyond context | Optional |
| `semantic_similarity` | **Meaning Match** | Cosine similarity between answer and ground truth embeddings | No |
| `rouge_l` | **Word Overlap** | Longest common subsequence F1 vs ground truth | No |

---

## Setup

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/your-username/rag-eval.git
cd rag-eval
pip install -r requirements.txt
```

### 2. Set environment variables

Create a `.env` file (never commit this):

```bash
GROQ_API_KEY=gsk_...          # https://console.groq.com
SUPABASE_URL=https://....supabase.co
SUPABASE_KEY=eyJ...            # anon/service role key
```

Load them before starting the server:

```bash
export $(cat .env | xargs)
```

### 3. Set up Supabase (for Document Q&A)

Run this SQL in your Supabase SQL Editor to create the vector table and search function:

```sql
-- Enable pgvector
create extension if not exists vector;

-- Documents table
create table documents (
  id bigint generated always as identity primary key,
  filename text not null,
  chunk_index int not null,
  content text not null,
  embedding vector(384),
  metadata jsonb default '{}'
);

-- Vector similarity search function
create or replace function match_documents(
  query_embedding vector(384),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  filename text,
  content text,
  similarity float
)
language sql stable
as $$
  select id, filename, content,
         1 - (embedding <=> query_embedding) as similarity
  from documents
  where 1 - (embedding <=> query_embedding) > match_threshold
  order by similarity desc
  limit match_count;
$$;
```

### 4. Build the React dashboard

```bash
cd dashboard
npm install
npm run build
cd ..
```

### 5. Start the server

```bash
python server.py
# Open http://localhost:8000
```

---

## Usage

### Web Dashboard

Open `http://localhost:8000` after starting the server. Three tabs:

- **Benchmark** — Run evaluations on built-in pipelines and datasets
- **Evaluate** — Test a single custom question/answer/context set
- **Documents** — Upload files, ask questions, and get scored answers

### CLI

```bash
# Run a benchmark
python benchmark.py run --pipeline naive --dataset science --max-samples 3

# Compare two pipelines
python benchmark.py compare --pipelines naive,semantic --dataset tech --max-samples 2

# Skip LLM metrics (fast, no API cost)
python benchmark.py run --pipeline naive --dataset all --no-llm

# List available metrics and datasets
python benchmark.py list-metrics
python benchmark.py list-datasets
```

### Programmatic API

```python
from core.models import RAGSample, BenchmarkConfig, MetricName
from core.evaluator import RAGEvaluator

samples = [
    RAGSample(
        question="What is photosynthesis?",
        ground_truth="Photosynthesis converts light energy to chemical energy in plants.",
        contexts=["Plants use sunlight, CO2, and water to produce glucose via photosynthesis."],
        answer="Photosynthesis is how plants convert sunlight into energy as glucose."
    )
]

config = BenchmarkConfig(
    name="my_eval",
    metrics=[MetricName.FAITHFULNESS, MetricName.GROUNDEDNESS, MetricName.ROUGE_L],
    use_llm_judge=True,
)

evaluator = RAGEvaluator(config)
report = evaluator.evaluate_dataset(samples, verbose=True)
evaluator.print_summary(report)
```

See `examples.py` for 5 detailed usage patterns (pipeline comparison, custom thresholds, single-sample analysis, and more).

---

## Supported File Types

Upload any of these to the Documents tab for Q&A:

| Type | Extensions |
| ---- | ---------- |
| PDF | `.pdf` |
| Word | `.docx`, `.doc` |
| Plain text | `.txt`, `.md` |
| YAML | `.yaml`, `.yml` |
| XML | `.xml` |
| JSON | `.json` |
| HTML | `.html`, `.htm` |
| Rich Text | `.rtf` |

---

## REST API

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/` | GET | Serve React dashboard |
| `/api/benchmark` | POST | Start a benchmark job |
| `/api/jobs` | GET | List all jobs |
| `/api/jobs/{id}` | GET | Poll job status |
| `/api/results/{id}` | GET | Get completed results |
| `/api/evaluate-sample` | POST | Evaluate a single sample |
| `/api/metrics` | GET | List all available metrics |
| `/api/datasets` | GET | List all built-in datasets |
| `/api/upload` | POST | Upload a document to Supabase |
| `/api/documents` | GET | List uploaded documents |
| `/api/documents/{filename}` | DELETE | Delete a document |
| `/api/ask` | POST | Ask a question against uploaded documents |

### Example: Ask a question via API

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the ESS algorithm?",
    "filename": "my_paper.pdf",
    "ground_truth": "optional — include to get evaluation scores"
  }'
```

---

## Built-in Datasets

| Dataset | Samples | Domains |
| ------- | ------- | ------- |
| `science` | 3 | biology, physics, geology |
| `tech` | 3 | ML, security, databases |
| `hallucination` | 2 | astronomy, history (tests hallucination resistance) |
| `all` | 8 | All of the above |

---

## Adding a Custom Metric

1. Create `metrics/your_metric.py` subclassing `BaseMetric`
2. Add `MetricName.YOUR_METRIC` to `core/models.py`
3. Register it in `metrics/__init__.py`

```python
from core.base_metric import BaseMetric
from core.models import RAGSample, MetricResult, MetricName

class YourMetric(BaseMetric):
    @property
    def name(self) -> MetricName:
        return MetricName.YOUR_METRIC

    def compute(self, sample: RAGSample) -> MetricResult:
        score = ...  # your logic
        return MetricResult(
            metric_name=self.name,
            score=score,
            explanation="...",
            threshold=self.threshold
        )
```

---

## What NOT to commit

The `.gitignore` already excludes:

- `.env` — API keys (Groq, Supabase)
- `venv/` — Python virtual environment
- `dashboard/node_modules/` — Node packages
- `dashboard/dist/` — Built React app (regenerate with `npm run build`)
- `uploaded_files/` — User-uploaded documents
- `reports/` — Generated evaluation JSON reports
- `__pycache__/` — Python bytecode
- `.DS_Store` — macOS metadata

---

## Design Philosophy

- **Hybrid retrieval**: Keyword scan (BM25-style) + pgvector similarity — robust even when embeddings are imperfect
- **Graceful degradation**: Every LLM metric falls back to a heuristic if the API fails
- **Pluggable pipelines**: Bring your own RAG pipeline via `BasePipeline`
- **Free to run**: Uses Groq's free tier (Llama 3.1 8B) for both generation and evaluation
