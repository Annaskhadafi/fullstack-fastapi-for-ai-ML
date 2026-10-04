import os
import asyncio
import re
from io import BytesIO
import time
import logging
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import httpx
from openai import AsyncOpenAI
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.models.document import Document
from app.schemas.rag import RetrievedChunk, RAGQueryResponse

logger = logging.getLogger(__name__)

_openai_client: Optional[AsyncOpenAI] = None
_local_embedding_model: Any = None
_provider = {"api_key": None, "model": None, "base_url": None}


def provider_settings() -> Dict[str, Any]:
    return {
        "configured": bool(_provider["api_key"] or settings.OPENAI_API_KEY),
        "model": _provider["model"] or settings.OPENAI_MODEL,
        "base_url": _provider["base_url"] or settings.OPENAI_BASE_URL or "https://api.openai.com/v1",
    }


def update_provider(api_key: str, model: str, base_url: str) -> Dict[str, Any]:
    _provider.update({"api_key": api_key.strip() or None, "model": model.strip() or None, "base_url": base_url.strip().rstrip("/") or None})
    return provider_settings()


def get_chat_client():
    key = _provider["api_key"] or settings.OPENAI_API_KEY
    if not key:
        return None
    return AsyncOpenAI(api_key=key, base_url=provider_settings()["base_url"])

LOCAL_DOCS_DIR = os.path.join("data", "documents")
LOCAL_DOCS_FILE = os.path.join(LOCAL_DOCS_DIR, "documents_registry.json")


def extract_pdf_text(file_bytes: bytes) -> Optional[str]:
    """Extract text locally so PDF ingestion needs no external API."""
    if not file_bytes:
        return None
    try:
        from pypdf import PdfReader
        pages = PdfReader(BytesIO(file_bytes)).pages
        text = "\n\n".join((page.extract_text() or "").strip() for page in pages)
        return text.strip() or None
    except Exception as exc:
        logger.warning("Local PDF extraction failed: %s", exc)
        return None


class UnifiedDocumentItem:
    """Wrapper ensuring both DB Documents and local document dicts work seamlessly in templates."""
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        self.id = str(data.get("id", uuid.uuid4().hex[:8]))
        self.title = str(data.get("title", "Dokumen"))
        self.content = str(data.get("content", ""))
        self.user_id = data.get("user_id")
        self.embedding = data.get("embedding")
        self.metadata_json = data.get("metadata_json", "{}")
        
        raw_dt = data.get("created_at")
        if isinstance(raw_dt, datetime):
            self.created_at = raw_dt
        elif isinstance(raw_dt, str):
            try:
                self.created_at = datetime.fromisoformat(raw_dt)
            except Exception:
                self.created_at = datetime.now(timezone.utc)
        else:
            self.created_at = datetime.now(timezone.utc)

    def __getattr__(self, item):
        return self._data.get(item)


def load_local_documents() -> List[Dict[str, Any]]:
    """Loads documents from data/documents/documents_registry.json."""
    if not os.path.exists(LOCAL_DOCS_FILE):
        return []
    try:
        with open(LOCAL_DOCS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"Could not read local document store: {e}")
        return []


def save_local_documents(docs: List[Dict[str, Any]]) -> None:
    """Saves documents to data/documents/documents_registry.json."""
    os.makedirs(LOCAL_DOCS_DIR, exist_ok=True)
    try:
        with open(LOCAL_DOCS_FILE, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write local document store: {e}")


async def delete_document(db: Optional[AsyncSession], document_id: str) -> bool:
    """Delete one document from the local fallback and database when available."""
    local_docs = load_local_documents()
    remaining = [doc for doc in local_docs if str(doc.get("id")) != str(document_id)]
    local_deleted = len(remaining) != len(local_docs)
    if local_deleted:
        save_local_documents(remaining)

    db_deleted = False
    if db is not None:
        try:
            result = await db.execute(delete(Document).where(Document.id == str(document_id)))
            await db.commit()
            db_deleted = bool(result.rowcount)
        except Exception as exc:
            logger.warning("Failed to delete document %s from database: %s", document_id, exc)
            await db.rollback()
    return local_deleted or db_deleted


def get_openai_client() -> Optional[AsyncOpenAI]:
    """
    Returns an AsyncOpenAI client instance.
    Supports official OpenAI, Cloudflare Workers AI, Groq, DeepSeek, and Ollama
    via OPENAI_BASE_URL and OPENAI_API_KEY.
    """
    global _openai_client
    if not settings.OPENAI_API_KEY:
        return None

    if _openai_client is None:
        kwargs: Dict[str, Any] = {"api_key": settings.OPENAI_API_KEY}
        if settings.OPENAI_BASE_URL:
            kwargs["base_url"] = settings.OPENAI_BASE_URL
        _openai_client = AsyncOpenAI(**kwargs)
        logger.info(f"Initialized OpenAI SDK client (base_url: {settings.OPENAI_BASE_URL or 'default'})")

    return _openai_client


def generate_local_embedding(text: str, dim: int = 768) -> List[float]:
    """
    Zero-dependency deterministic pseudo-embedding generator.
    Allows testing RAG vector storage and similarity search locally
    without requiring third-party API keys.
    """
    np.random.seed(int(hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()[:8], 16))
    vec = np.random.normal(0, 1, dim)
    norm = np.linalg.norm(vec)
    vec = (vec / norm).tolist() if norm > 0 else vec.tolist()
    return vec


async def get_embedding(text: str) -> List[float]:
    """
    Fetches vector embedding using OpenAI SDK (or compatible), Google Gemini, or local fallback.
    Returns 768-dimensional float vector.
    """
    clean_text = text.strip().replace("\n", " ")
    if not clean_text:
        return [0.0] * 768

    if settings.RAG_EMBEDDING_PROVIDER.lower() == "local":
        try:
            global _local_embedding_model
            if _local_embedding_model is None:
                from sentence_transformers import SentenceTransformer
                _local_embedding_model = SentenceTransformer(settings.RAG_LOCAL_EMBEDDING_MODEL)
            values = await asyncio.to_thread(_local_embedding_model.encode, clean_text, normalize_embeddings=True)
            values = np.asarray(values, dtype=float).reshape(-1).tolist()
            dim = settings.RAG_LOCAL_EMBEDDING_DIM
            return (values[:dim] + [0.0] * dim)[:dim]
        except Exception as e:
            logger.warning("Local embedding model unavailable; using deterministic fallback: %s", e)
            return generate_local_embedding(clean_text, dim=settings.RAG_LOCAL_EMBEDDING_DIM)

    # Optional remote embedding mode for legacy deployments.
    openai_client = get_openai_client()
    if openai_client:
        try:
            kwargs: Dict[str, Any] = {
                "model": settings.OPENAI_EMBEDDING_MODEL,
                "input": clean_text
            }
            if "text-embedding-3" in settings.OPENAI_EMBEDDING_MODEL:
                kwargs["dimensions"] = 768

            res = await openai_client.embeddings.create(**kwargs)
            emb = res.data[0].embedding
            if len(emb) == 768:
                return emb
            elif len(emb) > 768:
                return emb[:768]
            else:
                return emb + [0.0] * (768 - len(emb))
        except Exception as e:
            logger.warning(f"OpenAI SDK embedding failed: {e}")

    # 2. Google Gemini Embeddings (text-embedding-004: 768 dims)
    if settings.GEMINI_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={settings.GEMINI_API_KEY}"
            payload = {
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": clean_text}]}
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    values = data.get("embedding", {}).get("values", [])
                    if len(values) == 768:
                        return values
                else:
                    logger.warning(f"Gemini embedding API error: {res.text}")
        except Exception as e:
            logger.warning(f"Failed to call Gemini embedding API: {e}")

    # 3. Fallback local deterministic embedding
    return generate_local_embedding(clean_text, dim=768)


def chunk_text(text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
    """Split on paragraphs/sentences, then apply bounded overlap."""
    chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
    overlap = overlap if overlap is not None else settings.RAG_CHUNK_OVERLAP
    if len(text) <= chunk_size:
        return [text.strip()]
    units = [unit.strip() for unit in re.split(r"(?<=[.!?])\s+|\n\s*\n", text) if unit.strip()]
    chunks: List[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 1 > chunk_size:
            chunks.append(current.strip())
            tail = current[-overlap:].strip()
            current = f"{tail} {unit}" if tail else unit
        else:
            current = f"{current} {unit}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


async def get_recent_documents_unified(
    db: Optional[AsyncSession] = None,
    limit: int = 20
) -> List[UnifiedDocumentItem]:
    """
    Returns recent documents, merging DB records and local JSON document store.
    Never throws 500 even if the database is offline or uninitialized.
    """
    docs_map: Dict[str, UnifiedDocumentItem] = {}

    # 1. Load from local document store
    local_docs = load_local_documents()
    for d in local_docs:
        docs_map[d["id"]] = UnifiedDocumentItem(d)

    # 2. Load from database if reachable
    if db is not None:
        try:
            query = select(Document).order_by(Document.created_at.desc()).limit(limit)
            res = await db.execute(query)
            db_docs = res.scalars().all()
            for doc in db_docs:
                docs_map[doc.id] = UnifiedDocumentItem({
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "user_id": doc.user_id,
                    "embedding": doc.embedding,
                    "created_at": doc.created_at,
                    "source": "database"
                })
        except Exception as e:
            logger.warning(f"Could not load documents from DB (using local fallback): {e}")

    items = list(docs_map.values())
    items.sort(key=lambda x: x.created_at, reverse=True)
    return items[:limit]


async def ingest_document(
    db: Optional[AsyncSession],
    title: str,
    content: str,
    user_id: Optional[str] = None
) -> Tuple[List[UnifiedDocumentItem], bool]:
    """
    Chunks document content, calculates embeddings, persists to local document store,
    and syncs to PostgreSQL database if possible.
    Returns (created_documents, is_db_synced).
    """
    chunks = chunk_text(content)
    created_items: List[UnifiedDocumentItem] = []
    local_entries = []

    now_iso = datetime.now(timezone.utc).isoformat()

    for idx, chunk in enumerate(chunks):
        chunk_title = f"{title} (Part {idx + 1}/{len(chunks)})" if len(chunks) > 1 else title
        embedding = await get_embedding(chunk)
        doc_id = str(uuid.uuid4())

        entry = {
            "id": doc_id,
            "user_id": user_id,
            "title": chunk_title,
            "content": chunk,
            "embedding": embedding,
            "metadata_json": json.dumps({"original_title": title, "part": idx + 1, "total_parts": len(chunks)}),
            "created_at": now_iso,
            "source": "local_storage"
        }
        local_entries.append(entry)
        created_items.append(UnifiedDocumentItem(entry))

    # 1. Save locally (Guaranteed local fallback)
    existing_locals = load_local_documents()
    existing_locals.extend(local_entries)
    save_local_documents(existing_locals)

    # 2. Sync to DB if available
    db_synced = False
    if db is not None:
        try:
            from app.models.user import User
            valid_user_id = None
            if user_id:
                try:
                    u = await db.get(User, user_id)
                    if u:
                        valid_user_id = user_id
                except Exception:
                    valid_user_id = None

            for entry in local_entries:
                db_doc = Document(
                    id=entry["id"],
                    user_id=valid_user_id,
                    title=entry["title"],
                    content=entry["content"],
                    embedding=entry["embedding"],
                    metadata_json=entry["metadata_json"]
                )
                db.add(db_doc)
            await db.commit()
            db_synced = True
            logger.info(f"Ingested {len(chunks)} chunks into database.")
        except Exception as e:
            logger.warning(f"Failed to sync documents to DB (persisted locally): {e}")
            try:
                await db.rollback()
            except Exception:
                pass

    return created_items, db_synced


async def search_similar_documents(
    db: Optional[AsyncSession],
    query_text: str,
    top_k: int = 3
) -> List[RetrievedChunk]:
    """
    Performs cosine similarity search using pgvector on Postgres (if available),
    or in-memory numpy cosine similarity fallback across DB and local document store.
    """
    query_vec = await get_embedding(query_text)
    candidate_limit = max(top_k, settings.RAG_VECTOR_CANDIDATES)
    chunks_map: Dict[str, RetrievedChunk] = {}

    # 1. Try DB search if available
    if db is not None:
        if settings.is_postgres and settings.HAS_PGVECTOR:
            try:
                stmt = select(
                    Document,
                    Document.embedding.cosine_distance(query_vec).label("distance")
                ).order_by(Document.embedding.cosine_distance(query_vec)).limit(candidate_limit)

                result = await db.execute(stmt)
                for doc, distance in result.all():
                    sim = max(0.0, 1.0 - float(distance))
                    chunks_map[doc.id] = RetrievedChunk(
                        document_id=doc.id,
                        title=doc.title,
                        content=doc.content,
                        similarity_score=round(sim, 3)
                    )
            except Exception as e:
                logger.warning(f"Native pgvector search failed: {e}. Trying in-memory fallback.")

        # In-memory numpy fallback for DB docs if pgvector distance failed
        if not chunks_map:
            try:
                stmt = select(Document)
                res = await db.execute(stmt)
                all_db_docs = res.scalars().all()
                q_arr = np.array(query_vec)
                for d in all_db_docs:
                    if d.embedding is not None:
                        d_arr = np.array(d.embedding)
                        norm_q = np.linalg.norm(q_arr)
                        norm_d = np.linalg.norm(d_arr)
                        sim = float(np.dot(q_arr, d_arr) / (norm_q * norm_d)) if norm_q > 0 and norm_d > 0 else 0.0
                        chunks_map[d.id] = RetrievedChunk(
                            document_id=d.id,
                            title=d.title,
                            content=d.content,
                            similarity_score=round(max(0.0, sim), 3)
                        )
            except Exception as e:
                logger.warning(f"DB in-memory search failed: {e}")

    # 2. Local fallback document store similarity search
    local_docs = load_local_documents()
    if local_docs:
        q_arr = np.array(query_vec)
        for ld in local_docs:
            if ld["id"] not in chunks_map and ld.get("embedding"):
                d_arr = np.array(ld["embedding"])
                norm_q = np.linalg.norm(q_arr)
                norm_d = np.linalg.norm(d_arr)
                sim = float(np.dot(q_arr, d_arr) / (norm_q * norm_d)) if norm_q > 0 and norm_d > 0 else 0.0
                chunks_map[ld["id"]] = RetrievedChunk(
                    document_id=ld["id"],
                    title=ld["title"],
                    content=ld["content"],
                    similarity_score=round(max(0.0, sim), 3)
                )

    all_chunks = list(chunks_map.values())

    # Local BM25 reranking improves exact term and identifier matching after vector retrieval.
    query_terms = re.findall(r"[\w-]{2,}", query_text.lower())
    if all_chunks and query_terms:
        tokenized = [re.findall(r"[\w-]{2,}", item.content.lower()) for item in all_chunks]
        avg_len = sum(len(tokens) for tokens in tokenized) / max(len(tokenized), 1)
        doc_freq = {term: sum(term in set(tokens) for tokens in tokenized) for term in set(query_terms)}
        bm25_scores = []
        for tokens in tokenized:
            frequencies = {term: tokens.count(term) for term in set(query_terms)}
            score = 0.0
            for term, frequency in frequencies.items():
                if not frequency:
                    continue
                idf = np.log(1 + (len(tokenized) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                score += idf * (frequency * 2.0 / (frequency + 1.5 * (0.75 + 0.25 * len(tokens) / max(avg_len, 1))))
            bm25_scores.append(float(score))
        max_bm25 = max(bm25_scores, default=0.0) or 1.0
        reranked = []
        for item, bm25 in zip(all_chunks, bm25_scores):
            vector_score = float(item.similarity_score)
            item.similarity_score = round(0.65 * vector_score + 0.35 * (bm25 / max_bm25), 3)
            reranked.append(item)
        all_chunks = reranked

    all_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
    return all_chunks[:top_k]


async def answer_rag_query(
    db: Optional[AsyncSession],
    question: str,
    top_k: int = 3
) -> RAGQueryResponse:
    """Answers user question based on retrieved knowledge base context."""
    start_time = time.time()
    retrieved = await search_similar_documents(db, question, top_k=top_k)

    context_text = "\n\n".join([f"[{i+1}] {c.title}:\n{c.content}" for i, c in enumerate(retrieved)])

    model_used = "Local Context Synthesizer"
    answer = ""

    # 1. Try OpenAI SDK Client first if configured
    openai_client = get_chat_client()
    if openai_client:
        try:
            model_name = provider_settings()["model"]
            model_used = f"OpenAI SDK ({model_name})"
            system_prompt = "Kamu adalah asisten AI yang cerdas dan jujur. Jawab pertanyaan pengguna HANYA berdasarkan konteks dokumen yang diberikan. Jika jawaban tidak ditemukan, katakan dengan jelas."
            user_msg = f"Konteks Dokumen:\n{context_text if context_text else 'Tidak ada konteks dokumen yang relevan.'}\n\nPertanyaan:\n{question}\n\nJawaban:"

            response = await openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.3
            )
            answer = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"OpenAI SDK chat completion failed: {e}")

    # 2. Try Google Gemini if OpenAI was not used or failed
    if not answer and settings.GEMINI_API_KEY:
        try:
            model_used = "Google Gemini (gemini-1.5-flash)"
            prompt = f"""Kamu adalah asisten AI yang cerdas dan jujur. Jawab pertanyaan pengguna HANYA berdasarkan konteks dokumen yang diberikan berikut.

Konteks Dokumen:
{context_text if context_text else 'Tidak ada dokumen yang relevan ditemukan.'}

Pertanyaan:
{question}

Jawaban terstruktur:"""

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={settings.GEMINI_API_KEY}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    answer = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.warning(f"Gemini chat completion failed: {e}")

    # 3. Fallback synthesizer if no LLM API key
    if not answer:
        if not retrieved:
            answer = "Belum ada dokumen yang sesuai ditemukan di basis pengetahuan. Silakan upload dokumen referensi terlebih dahulu di tab dokumen."
        else:
            answer = f"Berdasarkan dokumen relevan yang ditemukan ({len(retrieved)} chunk):\n\n"
            for c in retrieved:
                answer += f"• **{c.title}** (relevansi {int(c.similarity_score * 100)}%):\n  \"{c.content[:200]}...\"\n\n"
            answer += "\n*(Tip: Masukkan OPENAI_API_KEY atau GEMINI_API_KEY di .env untuk mendapatkan jawaban naratif lengkap dari LLM)*"

    elapsed_ms = (time.time() - start_time) * 1000

    return RAGQueryResponse(
        question=question,
        answer=answer,
        retrieved_chunks=retrieved,
        model=model_used,
        execution_time_ms=round(elapsed_ms, 2)
    )
