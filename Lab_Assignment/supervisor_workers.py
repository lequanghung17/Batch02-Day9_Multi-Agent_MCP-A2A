"""Supervisor-Workers implementation for the Day08 RAG chatbot.

The original Day08 app is a RAG pipeline. This assignment reshapes it into
an agentic Supervisor-Workers pattern while keeping the implementation light
enough to run locally without extra services.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


STOPWORDS = {
    "và",
    "là",
    "của",
    "có",
    "cho",
    "về",
    "theo",
    "trong",
    "những",
    "nào",
    "gì",
    "được",
    "các",
    "một",
    "khi",
    "đến",
    "từ",
    "hoặc",
    "với",
    "hệ",
    "thống",
    "cần",
    "nêu",
    "này",
    "đó",
    "để",
}


@dataclass
class Evidence:
    content: str
    score: float
    metadata: dict
    worker: str

    def as_dict(self) -> dict:
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "worker": self.worker,
        }


class Worker(Protocol):
    name: str

    def run(self, question: str, top_k: int) -> list[Evidence]:
        """Return evidence for the question."""


def resolve_data_dir(candidate: Path | None = None) -> Path:
    """Resolve Day08 standardized data directory."""
    candidates = []
    if candidate is not None:
        candidates.append(candidate)
    candidates.extend(
        [
            Path("data") / "standardized",
            Path("..") / "Day08_RAG_pipeline_cohort2" / "data" / "standardized",
            Path(r"C:\Users\PC\Downloads\Day08_RAG_pipeline_cohort2\data\standardized"),
        ]
    )

    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved.exists() and any(resolved.rglob("*.md")):
            return resolved

    raise FileNotFoundError(
        "Could not find Day08 standardized markdown data. "
        "Pass --data-dir or set DAY08_DATA_DIR."
    )


def tokenize(text: str) -> set[str]:
    """Tokenize Vietnamese/English text with a small stopword list."""
    raw_tokens = re.findall(r"[\wÀ-ỹ]+", (text or "").lower(), flags=re.UNICODE)
    return {token for token in raw_tokens if token not in STOPWORDS and len(token) > 1}


def short_text(text: str, limit: int = 380) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0] + "..."


def load_corpus(data_dir: Path) -> list[dict]:
    """Load Day08 markdown files and split them into paragraph chunks."""
    corpus: list[dict] = []
    for md_file in data_dir.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = md_file.read_text(encoding="utf-8", errors="ignore")

        relative = md_file.relative_to(data_dir)
        doc_type = "legal" if "legal" in relative.parts else "news"
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 40]

        for index, paragraph in enumerate(paragraphs):
            corpus.append(
                {
                    "content": paragraph,
                    "metadata": {
                        "source": md_file.name,
                        "path": str(relative),
                        "type": doc_type,
                        "chunk_index": index,
                    },
                }
            )
    return corpus


def score_chunk(question: str, chunk: dict) -> float:
    """Simple lexical relevance score used by the workers."""
    query_tokens = tokenize(question)
    content_tokens = tokenize(chunk.get("content", ""))
    metadata = chunk.get("metadata", {}) or {}
    source_tokens = tokenize(metadata.get("source", ""))

    if not query_tokens or not content_tokens:
        return 0.0

    overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
    source_bonus = 0.25 if query_tokens & source_tokens else 0.0
    phrase_bonus = 0.15 if question.lower() in chunk.get("content", "").lower() else 0.0
    return overlap + source_bonus + phrase_bonus


class RetrievalWorker:
    """Base worker for retrieval over a document type."""

    name = "retrieval_worker"
    doc_type = "all"

    def __init__(self, corpus: list[dict]):
        self.corpus = corpus

    def run(self, question: str, top_k: int) -> list[Evidence]:
        scored: list[Evidence] = []
        for chunk in self.corpus:
            metadata = chunk.get("metadata", {}) or {}
            if self.doc_type != "all" and metadata.get("type") != self.doc_type:
                continue
            score = score_chunk(question, chunk)
            if score <= 0:
                continue
            scored.append(
                Evidence(
                    content=chunk["content"],
                    score=score,
                    metadata=metadata,
                    worker=self.name,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


class LegalDocumentWorker(RetrievalWorker):
    """Worker chuyên tìm evidence trong văn bản pháp luật."""

    name = "LegalDocumentWorker"
    doc_type = "legal"


class NewsContextWorker(RetrievalWorker):
    """Worker chuyên tìm evidence trong dữ liệu tin tức."""

    name = "NewsContextWorker"
    doc_type = "news"


class CitationAnswerWorker:
    """Worker tạo câu trả lời extractive có citation từ evidence."""

    name = "CitationAnswerWorker"

    def run(self, question: str, evidence: list[Evidence], top_k: int = 4) -> str:
        if not evidence:
            return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

        lines = [
            "Dựa trên các nguồn Day08 đã truy xuất, câu trả lời tóm tắt như sau:"
        ]
        for item in evidence[:top_k]:
            source = item.metadata.get("source", "unknown")
            lines.append(f"- {short_text(item.content)} [{source}]")

        lines.append(
            "Các trích dẫn trên lấy trực tiếp từ corpus Day08; cần đối chiếu source "
            "gốc khi dùng cho tư vấn pháp lý chính thức."
        )
        return "\n".join(lines)


class SupervisorAgent:
    """Supervisor điều phối các workers và tổng hợp final answer."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.corpus = load_corpus(data_dir)
        self.legal_worker = LegalDocumentWorker(self.corpus)
        self.news_worker = NewsContextWorker(self.corpus)
        self.answer_worker = CitationAnswerWorker()

    def run(self, question: str, top_k: int = 4) -> dict:
        trace: list[str] = []

        legal_evidence = self.legal_worker.run(question, top_k=top_k)
        trace.append(f"LegalDocumentWorker returned {len(legal_evidence)} chunks")

        news_evidence = self.news_worker.run(question, top_k=top_k)
        trace.append(f"NewsContextWorker returned {len(news_evidence)} chunks")

        merged = self._merge_evidence(legal_evidence + news_evidence, top_k=top_k)
        trace.append(f"Supervisor merged evidence to {len(merged)} chunks")

        answer = self.answer_worker.run(question, merged, top_k=top_k)
        trace.append("CitationAnswerWorker generated final answer")

        return {
            "question": question,
            "answer": answer,
            "sources": [item.as_dict() for item in merged],
            "trace": trace,
            "pattern": "Supervisor-Workers",
            "workers": [
                self.legal_worker.name,
                self.news_worker.name,
                self.answer_worker.name,
            ],
        }

    @staticmethod
    def _merge_evidence(evidence: list[Evidence], top_k: int) -> list[Evidence]:
        """Deduplicate by source/chunk and keep highest-scoring evidence."""
        by_key: dict[tuple[str, int], Evidence] = {}
        for item in evidence:
            key = (
                item.metadata.get("source", "unknown"),
                int(item.metadata.get("chunk_index", -1)),
            )
            if key not in by_key or item.score > by_key[key].score:
                by_key[key] = item

        merged = list(by_key.values())
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:top_k]

