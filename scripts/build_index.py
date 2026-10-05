from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "knowledge_base"
INDEX = ROOT / "index"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 3
DEMO = [
    "Кто такой Ксарн Велгор?",
    "Как называется столица планеты Ти'лора?",
    "От чего питается экспериментальный HyperRelay?",
]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, 1e-12, None)).astype("float32")


def load_chunks() -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    for path in sorted(KB.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = text.splitlines()[0].lstrip("# ").strip()
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        for i, piece in enumerate(splitter.split_text(text)):
            start = text.find(piece[:80]) if piece else 0
            chunks.append(
                {
                    "chunk_id": f"{path.stem}-{i:03d}",
                    "text": piece,
                    "source": rel,
                    "title": title,
                    "chunk_index": i,
                    "start_index": max(start, 0),
                }
            )
    return chunks


def embed(model: TextEmbedding, texts: list[str]) -> np.ndarray:
    vectors = np.vstack(list(model.embed(texts))).astype("float32")
    return l2_normalize(vectors)


def build() -> tuple[faiss.Index, list[dict], TextEmbedding, dict]:
    started = time.perf_counter()
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("knowledge_base пуста. Сначала python scripts/prepare_kb.py")
    model = TextEmbedding(model_name=MODEL)
    vectors = embed(model, [c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    INDEX.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX / "faiss.index"))
    (INDEX / "chunks.pkl").write_bytes(pickle.dumps(chunks))
    stats = {
        "embedding_model": MODEL,
        "model_url": "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "embedding_dim": int(vectors.shape[1]),
        "vector_db": "FAISS IndexFlatIP",
        "knowledge_base": "knowledge_base/",
        "documents": len({c["source"] for c in chunks}),
        "chunks": len(chunks),
        "chunk_size_chars": CHUNK_SIZE,
        "chunk_overlap_chars": CHUNK_OVERLAP,
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }
    (INDEX / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return index, chunks, model, stats


def load_index():
    index_path = INDEX / "faiss.index"
    meta_path = INDEX / "chunks.pkl"
    if not index_path.exists() or not meta_path.exists():
        raise SystemExit("Индекс не найден. Сначала python scripts/build_index.py")
    index = faiss.read_index(str(index_path))
    chunks = pickle.loads(meta_path.read_bytes())
    model = TextEmbedding(model_name=MODEL)
    return index, chunks, model


def search(index, chunks: list[dict], model: TextEmbedding, query: str) -> list[tuple[float, dict]]:
    scores, ids = index.search(embed(model, [query]), TOP_K)
    return [
        (float(score), chunks[int(idx)])
        for score, idx in zip(scores[0], ids[0])
        if int(idx) >= 0
    ]


def show(query: str, hits: list[tuple[float, dict]]) -> None:
    print(f"\nЗапрос: {query}")
    for score, item in hits:
        snippet = " ".join(item["text"].split())
        print(
            f"- {score:.3f} {item['title']} [{item['source']}] "
            f"id={item['chunk_id']} pos={item['start_index']}"
        )
        print(f"  {snippet}")


def main() -> None:
    index, chunks, model, stats = build()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    for query in (sys.argv[1:] or DEMO):
        show(query, search(index, chunks, model, query))


if __name__ == "__main__":
    main()
