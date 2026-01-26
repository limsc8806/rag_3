from __future__ import annotations

from typing import List

from ..index.stores import IndexBundle
from ..types import RetrievalResult


def retrieve(
    bundle: IndexBundle,
    query: str,
    *,
    top_k_text: int = 6,
    top_k_img: int = 4,
    top_k_table: int = 4,
    merge_top_k: int = 8,
    weight_text: float = 1.0,
    weight_table: float = 1.0,
    weight_image: float = 1.0,
) -> List[RetrievalResult]:
    text_results = bundle.text.search(query, top_k=top_k_text) if top_k_text else []
    img_results = (
        bundle.image.search(query, top_k=top_k_img)
        if top_k_img and bundle.image is not None
        else []
    )
    table_results = (
        bundle.table.search(query, top_k=top_k_table)
        if top_k_table and bundle.table is not None
        else []
    )

    def _apply_weight(results: List[RetrievalResult], weight: float) -> List[RetrievalResult]:
        if weight == 1.0:
            return results
        weighted: List[RetrievalResult] = []
        for r in results:
            weighted.append(RetrievalResult(document=r.document, score=r.score * weight))
        return weighted

    merged = (
        _apply_weight(text_results, weight_text)
        + _apply_weight(table_results, weight_table)
        + _apply_weight(img_results, weight_image)
    )
    merged.sort(key=lambda r: r.score, reverse=True)
    return merged[:merge_top_k]
