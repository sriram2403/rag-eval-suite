"""
Pluggable RAG Pipeline Interface + example implementations.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import os


class BasePipeline(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def retrieve(self, question: str) -> List[str]:
        pass

    @abstractmethod
    def generate(self, question: str, contexts: List[str]) -> str:
        pass

    def run(self, question: str) -> Tuple[str, List[str]]:
        contexts = self.retrieve(question)
        answer = self.generate(question, contexts)
        return answer, contexts


class NaiveRAGPipeline(BasePipeline):
    def __init__(self, documents: List[str], top_k: int = 3):
        self.documents = documents
        self.top_k = top_k
        self._client = None
        self._vectorizer = None
        self._doc_vectors = None
        self._fit()

    @property
    def name(self) -> str:
        return "naive_rag_tfidf"

    def _fit(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                ngram_range=(1, 2)
            )
            self._doc_vectors = self._vectorizer.fit_transform(self.documents)
        except ImportError:
            pass

    def _get_client(self):
        if self._client is None:
            if os.environ.get("GROQ_API_KEY"):
                from groq_client import GroqClient
                self._client = GroqClient()
            else:
                import anthropic
                self._client = anthropic.Anthropic()
        return self._client

    def retrieve(self, question: str) -> List[str]:
        if self._vectorizer is not None:
            try:
                import numpy as np
                from sklearn.metrics.pairwise import cosine_similarity
                q_vec = self._vectorizer.transform([question])
                sims = cosine_similarity(q_vec, self._doc_vectors)[0]
                top_indices = np.argsort(sims)[-self.top_k:][::-1]
                return [self.documents[i] for i in top_indices if sims[i] > 0]
            except Exception:
                pass
        q_words = set(question.lower().split())
        scored = [
            (sum(1 for w in q_words if w in doc.lower()), doc)
            for doc in self.documents
        ]
        scored.sort(reverse=True)
        return [doc for _, doc in scored[:self.top_k] if _ > 0]

    def generate(self, question: str, contexts: List[str]) -> str:
        client = self._get_client()
        context_text = "\n\n".join(
            f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)
        )
        prompt = f"""You are a helpful assistant that answers questions using only the provided context.

Instructions:
- Give a complete, informative answer in 2-3 sentences
- Include specific details like dates, names, and locations from the context
- Do not add any information that is not in the context
- Do not mention the context in your answer, just answer naturally
- Only say you cannot answer if the information is truly not present in the context

Context:
{context_text}

Question: {question}

Answer:"""
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()


class SemanticRAGPipeline(BasePipeline):
    def __init__(self, documents: List[str], top_k: int = 3,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.documents = documents
        self.top_k = top_k
        self.embedding_model_name = embedding_model
        self._client = None
        self._embedder = None
        self._doc_embeddings = None

    @property
    def name(self) -> str:
        return "semantic_rag_embeddings"

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
            self._doc_embeddings = self._embedder.encode(
                self.documents, show_progress_bar=False
            )
        return self._embedder

    def _get_client(self):
        if self._client is None:
            if os.environ.get("GROQ_API_KEY"):
                from groq_client import GroqClient
                self._client = GroqClient()
            else:
                import anthropic
                self._client = anthropic.Anthropic()
        return self._client

    def retrieve(self, question: str) -> List[str]:
        try:
            import numpy as np
            embedder = self._get_embedder()
            q_emb = embedder.encode([question])
            doc_norms = np.linalg.norm(self._doc_embeddings, axis=1, keepdims=True)
            q_norm = np.linalg.norm(q_emb)
            sims = (self._doc_embeddings @ q_emb.T).flatten() / (doc_norms.flatten() * q_norm + 1e-8)
            top_indices = np.argsort(sims)[-self.top_k:][::-1]
            return [self.documents[i] for i in top_indices]
        except Exception:
            q_words = set(question.lower().split())
            scored = [
                (sum(1 for w in q_words if w in doc.lower()), doc)
                for doc in self.documents
            ]
            scored.sort(reverse=True)
            return [doc for _, doc in scored[:self.top_k]]

    def generate(self, question: str, contexts: List[str]) -> str:
        client = self._get_client()
        context_text = "\n\n".join(
            f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)
        )
        prompt = f"""You are a helpful assistant that answers questions using only the provided context.

Instructions:
- Give a complete, informative answer in 2-3 sentences
- Include specific details like dates, names, and locations from the context
- Do not add any information that is not in the context
- Do not mention the context in your answer, just answer naturally
- Only say you cannot answer if the information is truly not present in the context

Context:
{context_text}

Question: {question}

Answer:"""
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()


class MockPipeline(BasePipeline):
    def __init__(self, answers: dict = None):
        self.answers = answers or {}

    @property
    def name(self) -> str:
        return "mock_pipeline"

    def retrieve(self, question: str) -> List[str]:
        return self.answers.get(question, {}).get("contexts", [
            "This is a mock context for testing purposes.",
            "Another mock context with relevant information."
        ])

    def generate(self, question: str, contexts: List[str]) -> str:
        return self.answers.get(question, {}).get(
            "answer", "This is a mock answer for testing."
        )
