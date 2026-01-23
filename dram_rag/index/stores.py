from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import warnings
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..types import Document, RetrievalResult


class TfidfIndex:
    """Minimal persistent TF-IDF index for Documents."""

    def __init__(self, name: str):
        self.name = name
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix: Optional[sparse.csr_matrix] = None
        self.documents: List[Document] = []

    def build(self, documents: List[Document], *, min_df: int = 1, max_df: float = 1.0) -> None:
        self.documents = documents
        if not documents:
            self.vectorizer = None
            self.matrix = sparse.csr_matrix((0, 0))
            return
        texts = [d.text for d in documents]
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        docs_path = index_dir / f"{self.name}.docs.jsonl"
        with docs_path.open("w", encoding="utf-8") as f:
            for d in self.documents:
                rec = {"text": d.text, "metadata": d.metadata, "doc_type": d.doc_type}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if not self.documents:
            vec_path = index_dir / f"{self.name}.vectorizer.joblib"
            mat_path = index_dir / f"{self.name}.matrix.npz"
            if vec_path.exists():
                vec_path.unlink()
            if mat_path.exists():
                mat_path.unlink()
            return

        if self.vectorizer is None or self.matrix is None:
            raise ValueError("Index is not built.")

        joblib.dump(self.vectorizer, index_dir / f"{self.name}.vectorizer.joblib")
        sparse.save_npz(index_dir / f"{self.name}.matrix.npz", self.matrix)

    @classmethod
    def load(cls, name: str, index_dir: str | Path) -> "TfidfIndex":
        index_dir = Path(index_dir)
        obj = cls(name)

        vec_path = index_dir / f"{name}.vectorizer.joblib"
        mat_path = index_dir / f"{name}.matrix.npz"
        docs_path = index_dir / f"{name}.docs.jsonl"

        if not docs_path.exists():
            raise FileNotFoundError(
                f"Missing index files for '{name}' under {index_dir}. "
                "Run build_index first."
            )

        if not vec_path.exists() or not mat_path.exists():
            obj.documents = []
            with docs_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    obj.documents.append(
                        Document(
                            text=rec["text"],
                            metadata=rec.get("metadata", {}),
                            doc_type=rec.get("doc_type", "text"),
                        )
                    )
            obj.vectorizer = None
            obj.matrix = sparse.csr_matrix((0, 0))
            return obj

        obj.vectorizer = joblib.load(vec_path)
        obj.matrix = sparse.load_npz(mat_path)

        docs: List[Document] = []
        with docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                docs.append(Document(text=rec["text"], metadata=rec.get("metadata", {}), doc_type=rec.get("doc_type", "text")))
        obj.documents = docs
        return obj

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if self.vectorizer is None or self.matrix is None:
            return []
        if not self.documents:
            return []

        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix).ravel()
        if top_k <= 0:
            top_k = 5

        idx = np.argsort(-sims)[: min(top_k, len(sims))]
        results: List[RetrievalResult] = []
        for i in idx:
            results.append(RetrievalResult(document=self.documents[int(i)], score=float(sims[int(i)])))
        return results


class IndexBundle:
    """Convenience wrapper: separate indices for text, image, and table docs."""

    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.text = TfidfIndex("text")
        self.image: Optional[TfidfIndex] = TfidfIndex("image")
        self.table: Optional[TfidfIndex] = TfidfIndex("table")

    def build_and_save(
        self,
        text_docs: List[Document],
        image_docs: List[Document],
        table_docs: List[Document],
    ) -> None:
        self._write_metadata()
        self.text.build(text_docs)
        self.image.build(image_docs)
        self.table.build(table_docs)
        self.text.save(self.index_dir)
        self.image.save(self.index_dir)
        self.table.save(self.index_dir)

    @classmethod
    def load(cls, index_dir: str | Path) -> "IndexBundle":
        obj = cls(index_dir)
        obj._check_metadata()
        obj.text = TfidfIndex.load("text", index_dir)
        image_docs = Path(index_dir) / "image.docs.jsonl"
        table_docs = Path(index_dir) / "table.docs.jsonl"
        obj.image = TfidfIndex.load("image", index_dir) if image_docs.exists() else None
        obj.table = TfidfIndex.load("table", index_dir) if table_docs.exists() else None
        return obj

    def _write_metadata(self) -> None:
        try:
            import sklearn  # type: ignore
        except Exception:
            return
        meta = {
            "sklearn_version": getattr(sklearn, "__version__", ""),
        }
        path = self.index_dir / "index_metadata.json"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def _check_metadata(self) -> None:
        path = self.index_dir / "index_metadata.json"
        if not path.exists():
            return
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        try:
            import sklearn  # type: ignore
        except Exception:
            return
        saved = (meta.get("sklearn_version") or "").strip()
        current = getattr(sklearn, "__version__", "")
        if saved and current and saved != current:
            warnings.warn(
                f"Index built with scikit-learn {saved}, running {current}. "
                "Rebuild index for compatibility.",
                UserWarning,
            )
