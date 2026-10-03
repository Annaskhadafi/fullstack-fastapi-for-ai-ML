import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.core.config import settings
from app.services.rag_service import get_openai_client

STORE = os.path.join("data", "documents", "vectorless_registry.json")


def _load() -> List[Dict[str, Any]]:
    try:
        with open(STORE, "r", encoding="utf-8") as handle:
            value = json.load(handle)
            return value if isinstance(value, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save(items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    with open(STORE, "w", encoding="utf-8") as handle:
        json.dump(items[-1000:], handle, ensure_ascii=False, indent=2)


def _chunks(content: str, size: int = 900) -> List[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
    chunks: List[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= size:
            chunks.append(paragraph)
        else:
            chunks.extend(paragraph[i:i + size].strip() for i in range(0, len(paragraph), size))
    return chunks or [content.strip()]


def add_document(title: str, content: str, user_id: str | None = None) -> int:
    items = _load()
    now = datetime.now(timezone.utc).isoformat()
    entries = [{"id": str(uuid.uuid4()), "title": title, "content": chunk, "created_at": now, "user_id": user_id}
               for chunk in _chunks(content)]
    items.extend(entries)
    _save(items)
    return len(entries)


def list_documents(limit: int = 100) -> List[Dict[str, Any]]:
    return list(reversed(_load()))[:limit]


def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    terms = set(re.findall(r"[\w-]{2,}", query.lower()))
    ranked = []
    for item in _load():
        words = set(re.findall(r"[\w-]{2,}", item["content"].lower()))
        overlap = terms & words
        if overlap:
            score = len(overlap) / max(len(terms), 1)
            ranked.append({**item, "score": round(score, 4), "matched_terms": sorted(overlap)})
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:top_k]


async def answer(query: str, top_k: int = 5) -> Dict[str, Any]:
    matches = search(query, top_k)
    context = "\n\n".join(f"[{item['title']}]\n{item['content']}" for item in matches)
    text = ""
    model = "Lexical Context (Vectorless)"
    client = get_openai_client()
    if client and context:
        try:
            response = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "system", "content": "Jawab hanya dari konteks dokumen. Jika tidak ada jawabannya, katakan tidak ditemukan."},
                          {"role": "user", "content": f"Konteks:\n{context}\n\nPertanyaan: {query}"}],
                temperature=0.2,
            )
            text = response.choices[0].message.content or ""
            model = f"{model} + {settings.OPENAI_MODEL}"
        except Exception:
            text = ""
    if not text:
        text = "Belum ada jawaban dari indeks teks." if not matches else "\n\n".join(
            f"**{item['title']}** ({round(item['score'] * 100)}% cocok)\n{item['content'][:500]}"
            for item in matches
        )
    return {"answer": text, "model": model, "matches": matches}
