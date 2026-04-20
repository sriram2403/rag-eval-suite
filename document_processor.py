"""
Document Processor — extracts text from PDF/DOC files,
chunks it, embeds it, and stores in Supabase.
"""
import os
import re
from typing import List, Dict
from supabase import create_client, Client


def get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {e}")


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a Word document."""
    try:
        from docx import Document
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to read DOCX: {e}")


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a plain text file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read().strip()


def extract_text_from_yaml(file_path: str) -> str:
    import yaml
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        data = yaml.safe_load(f)
    return yaml.dump(data, default_flow_style=False, allow_unicode=True).strip()


def extract_text_from_xml(file_path: str) -> str:
    import xml.etree.ElementTree as ET
    tree = ET.parse(file_path)
    parts = []
    for elem in tree.iter():
        if elem.text and elem.text.strip():
            parts.append(elem.text.strip())
        if elem.tail and elem.tail.strip():
            parts.append(elem.tail.strip())
    return '\n'.join(parts).strip()


def extract_text_from_json(file_path: str) -> str:
    import json
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)
    return json.dumps(data, indent=2, ensure_ascii=False).strip()


def extract_text_from_html(file_path: str) -> str:
    from html.parser import HTMLParser
    class _Strip(HTMLParser):
        def __init__(self):
            super().__init__()
            self._parts = []
            self._skip = False
        def handle_starttag(self, tag, attrs):
            if tag in ('script', 'style'):
                self._skip = True
        def handle_endtag(self, tag):
            if tag in ('script', 'style'):
                self._skip = False
        def handle_data(self, data):
            if not self._skip and data.strip():
                self._parts.append(data.strip())
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    parser = _Strip()
    parser.feed(raw)
    return '\n'.join(parser._parts).strip()


def extract_text_from_rtf(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    # Strip RTF control words, groups and special characters
    text = re.sub(r'\\[a-z]+[-]?\d*[ ]?', ' ', raw)
    text = re.sub(r'[{}\\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text(file_path: str) -> str:
    """Auto-detect file type and extract text."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif ext in ['.txt', '.md']:
        return extract_text_from_txt(file_path)
    elif ext in ['.yaml', '.yml']:
        return extract_text_from_yaml(file_path)
    elif ext == '.xml':
        return extract_text_from_xml(file_path)
    elif ext == '.json':
        return extract_text_from_json(file_path)
    elif ext in ['.html', '.htm']:
        return extract_text_from_html(file_path)
    elif ext == '.rtf':
        return extract_text_from_rtf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """
    Smart chunking that respects sentence boundaries.
    Prevents cutting mid-sentence which destroys context.
    """
    import re

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # If adding this sentence exceeds chunk size, save current chunk
        if current_word_count + sentence_words > chunk_size and current_chunk:
            chunk_text_val = ' '.join(current_chunk)
            if len(chunk_text_val.strip()) > 50:
                chunks.append(chunk_text_val.strip())

            # Keep last few sentences for overlap
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_chunk):
                overlap_count += len(s.split())
                if overlap_count > overlap:
                    break
                overlap_sentences.insert(0, s)

            current_chunk = overlap_sentences
            current_word_count = sum(len(s.split()) for s in current_chunk)

        current_chunk.append(sentence)
        current_word_count += sentence_words

    # Add final chunk
    if current_chunk:
        chunk_text_val = ' '.join(current_chunk)
        if len(chunk_text_val.strip()) > 50:
            chunks.append(chunk_text_val.strip())

    return chunks


def get_embedder():
    """Get sentence transformer embedder."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception:
        return None


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Convert texts to embedding vectors."""
    embedder = get_embedder()
    if embedder:
        embeddings = embedder.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    else:
        # Fallback: simple hash-based fake embeddings (for testing)
        import hashlib
        import math
        result = []
        for text in texts:
            h = hashlib.md5(text.encode()).hexdigest()
            vec = [(int(h[i:i+2], 16) / 255.0) for i in range(0, min(len(h), 48), 2)]
            # Pad to 384 dimensions
            while len(vec) < 384:
                vec.append(0.0)
            result.append(vec[:384])
        return result


def upload_document(file_path: str, filename: str = None) -> Dict:
    """
    Full pipeline: extract → chunk → embed → store in Supabase.
    Returns summary of what was uploaded.
    """
    if not filename:
        filename = os.path.basename(file_path)

    print(f"  Extracting text from {filename}...")
    text = extract_text(file_path)
    print(f"  Extracted {len(text)} characters")

    print(f"  Chunking text...")
    chunks = chunk_text(text)
    print(f"  Created {len(chunks)} chunks")

    print(f"  Generating embeddings...")
    embeddings = embed_texts(chunks)
    print(f"  Generated {len(embeddings)} embeddings")

    print(f"  Storing in Supabase...")
    supabase = get_supabase()

    # Delete existing chunks for this file
    supabase.table('documents').delete().eq('filename', filename).execute()

    # Insert new chunks in batches
    batch_size = 50
    total_inserted = 0

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]

        rows = [
            {
                'filename': filename,
                'chunk_index': i + j,
                'content': chunk,
                'embedding': embedding,
                'metadata': {
                    'chunk_index': i + j,
                    'total_chunks': len(chunks),
                    'file_type': os.path.splitext(filename)[1]
                }
            }
            for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings))
        ]

        supabase.table('documents').insert(rows).execute()
        total_inserted += len(rows)
        print(f"  Inserted batch {i//batch_size + 1} ({total_inserted}/{len(chunks)} chunks)")

    print(f"  Done! {filename} uploaded successfully")
    return {
        'filename': filename,
        'total_chunks': len(chunks),
        'total_characters': len(text)
    }


def list_documents() -> List[Dict]:
    """List all uploaded documents."""
    supabase = get_supabase()
    result = supabase.table('documents')\
        .select('filename, chunk_index, metadata')\
        .order('filename')\
        .execute()

    # Group by filename
    files = {}
    for row in (result.data or []):
        fname = row['filename']
        if fname not in files:
            files[fname] = {'filename': fname, 'chunks': 0}
        files[fname]['chunks'] += 1

    return list(files.values())


def delete_document(filename: str) -> bool:
    """Delete all chunks for a document."""
    supabase = get_supabase()
    supabase.table('documents').delete().eq('filename', filename).execute()
    return True


def fetch_all_chunks(filename: str = None) -> List[str]:
    """Fetch every stored chunk (optionally filtered by filename)."""
    supabase = get_supabase()
    q = supabase.table('documents').select('content, filename').order('chunk_index')
    if filename:
        q = q.eq('filename', filename)
    result = q.execute()
    return [r['content'] for r in (result.data or [])]


def keyword_rank(query: str, chunks: List[str], top_k: int = 6) -> List[str]:
    """Pure-Python BM25-style keyword ranking over a list of chunks."""
    import re
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in",
                 "to", "and", "or", "for", "on", "at", "by", "it", "this"}
    q_tokens = {
        re.sub(r'[^\w]', '', w).lower()
        for w in query.split()
        if len(re.sub(r'[^\w]', '', w)) > 2
    } - stopwords

    scored = []
    for chunk in chunks:
        c_tokens = {
            re.sub(r'[^\w]', '', w).lower()
            for w in chunk.split()
        }
        overlap = len(q_tokens & c_tokens)
        scored.append((overlap, chunk))

    scored.sort(reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def search_documents_hybrid(
    query: str,
    filename: str = None,
    top_k: int = 6,
    threshold: float = 0.2
) -> List[str]:
    """
    Hybrid search: combine vector similarity + keyword matching.
    Better recall than either approach alone.
    """
    supabase = get_supabase()

    # 1. Vector search
    query_embedding = embed_texts([query])[0]
    vector_results = supabase.rpc(
        'match_documents',
        {
            'query_embedding': query_embedding,
            'match_threshold': threshold,
            'match_count': top_k * 2  # get more candidates
        }
    ).execute()

    vector_chunks = {
        r['content']: r['similarity']
        for r in (vector_results.data or [])
        if not filename or r['filename'] == filename
    }

    # 2. Keyword search using PostgreSQL full-text search
    import re as _re
    query_words = ' | '.join(
        w for w in (_re.sub(r'[^\w]', '', tok) for tok in query.split())
        if len(w) > 3
    )

    try:
        keyword_results = supabase.table('documents')\
            .select('content, filename')\
            .text_search('content', query_words)\
            .limit(top_k * 2)\
            .execute()

        keyword_chunks = {
            r['content']: 0.5  # fixed score for keyword matches
            for r in (keyword_results.data or [])
            if not filename or r['filename'] == filename
        }
    except:
        keyword_chunks = {}

    # 3. Combine scores — balanced weights so keyword search can surface relevant
    # chunks even when vector embeddings are poor quality (e.g. NumPy compat issues)
    all_chunks = {}
    for content, score in vector_chunks.items():
        all_chunks[content] = all_chunks.get(content, 0) + score * 0.5

    for content, score in keyword_chunks.items():
        all_chunks[content] = all_chunks.get(content, 0) + score * 0.5

    # If keyword search found chunks not in vector results, boost them to top
    # so they don't get buried by bad vector scores
    keyword_only = set(keyword_chunks) - set(vector_chunks)
    for content in keyword_only:
        all_chunks[content] = all_chunks.get(content, 0) + 0.2

    # Sort by combined score
    ranked = sorted(all_chunks.items(), key=lambda x: x[1], reverse=True)
    return [content for content, _ in ranked[:top_k]]