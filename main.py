import os
from typing import List, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Lazy imports to avoid startup crashes if heavy deps are still installing

def _np():
    try:
        import numpy as _numpy  # type: ignore
        return _numpy
    except Exception as e:
        raise RuntimeError(f"NumPy not available: {e}")

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore
    HAS_FAISS = False

from sentence_transformers import SentenceTransformer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_embedder: Optional[SentenceTransformer] = None

# In-memory corpus and vector store
DOCUMENTS: List[str] = []
EMB_MATRIX = None  # type: ignore  # Will hold numpy ndarray at runtime
FAISS_INDEX = None
EMB_DIM: Optional[int] = None

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


class IngestResponse(BaseModel):
    chunks: int

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

class QueryAnswer(BaseModel):
    answer: str
    contexts: List[str]
    scores: List[float]


@app.get("/health")
def health():
    try:
        np = _np()
        np_ok = True
        np_ver = str(np.__version__)
    except Exception as e:
        np_ok = False
        np_ver = str(e)
    return {
        "status": "ok",
        "numpy": {"available": np_ok, "version": np_ver},
        "faiss": HAS_FAISS,
        "docs": len(DOCUMENTS),
    }


def get_embedder() -> SentenceTransformer:
    global _embedder, EMB_DIM
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)
        test = _embedder.encode(["hello"], convert_to_numpy=True)
        EMB_DIM = int(test.shape[1])
    return _embedder


def ensure_faiss_index() -> None:
    global FAISS_INDEX, EMB_DIM
    if not HAS_FAISS:
        return
    if FAISS_INDEX is None:
        if EMB_DIM is None:
            get_embedder()
        FAISS_INDEX = faiss.IndexFlatIP(int(EMB_DIM))


def add_vectors_to_index(vectors) -> None:  # vectors is numpy ndarray
    global EMB_MATRIX, FAISS_INDEX
    np = _np()
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    EMB_MATRIX = vectors if EMB_MATRIX is None else np.vstack([EMB_MATRIX, vectors])
    if HAS_FAISS:
        ensure_faiss_index()
        FAISS_INDEX.add(vectors)


def search_vectors(query_vec, top_k: int) -> Tuple[List[int], List[float]]:
    np = _np()
    if HAS_FAISS and FAISS_INDEX is not None and (FAISS_INDEX.ntotal > 0):
        q = query_vec.astype(np.float32)[None, :]
        D, I = FAISS_INDEX.search(q, top_k)
        return I[0].tolist(), [float(x) for x in D[0].tolist()]
    if EMB_MATRIX is None or EMB_MATRIX.shape[0] == 0:
        return [], []
    sims = EMB_MATRIX @ query_vec
    idxs = np.argsort(-sims)[:top_k]
    return idxs.tolist(), [float(sims[i]) for i in idxs]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + size)
        chunk = " ".join(tokens[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(tokens):
            break
        start = max(0, end - overlap)
    return chunks


@app.get("/")
def root():
    return {"message": f"RAG backend running ({'FAISS' if HAS_FAISS else 'NumPy'} index)"}


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_files(files: List[UploadFile] = File(...)):
    global DOCUMENTS
    contents = []
    for f in files:
        data = await f.read()
        text = None
        name = (f.filename or "").lower()
        if name.endswith(".txt") or name.endswith(".md"):
            text = data.decode("utf-8", errors="ignore")
        elif name.endswith(".pdf"):
            from pdfminer.high_level import extract_text
            import io
            text = extract_text(io.BytesIO(data))
        else:
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                pass
        if not text:
            raise HTTPException(status_code=400, detail=f"Unsupported file: {f.filename}")
        contents.append(text)

    # Chunk
    all_chunks: List[str] = []
    for content in contents:
        all_chunks.extend(chunk_text(content))

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No text chunks produced from uploads")

    # Embed and add to index
    embedder = get_embedder()
    vecs = embedder.encode(all_chunks, convert_to_numpy=True, normalize_embeddings=True)
    add_vectors_to_index(vecs)

    DOCUMENTS.extend(all_chunks)

    return IngestResponse(chunks=len(all_chunks))


@app.post("/api/query", response_model=QueryAnswer)
async def query_rag(payload: QueryRequest):
    if len(DOCUMENTS) == 0:
        raise HTTPException(status_code=400, detail="No documents ingested yet")

    embedder = get_embedder()
    q_vec = embedder.encode([payload.question], convert_to_numpy=True, normalize_embeddings=True)[0]
    top_k = max(1, min(int(payload.top_k), len(DOCUMENTS)))

    idxs, sims = search_vectors(q_vec, top_k)
    if not idxs:
        raise HTTPException(status_code=500, detail="Search failed")

    contexts = [DOCUMENTS[i] for i in idxs]
    answer = contexts[0] if contexts else ""

    return QueryAnswer(answer=answer, contexts=contexts, scores=sims)


@app.post("/api/reset")
async def reset_index():
    global DOCUMENTS, EMB_MATRIX, FAISS_INDEX
    DOCUMENTS = []
    EMB_MATRIX = None
    FAISS_INDEX = None
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
