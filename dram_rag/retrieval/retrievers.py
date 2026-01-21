from __future__ import annotations

from typing import List

from ..index.stores import IndexBundle
from ..types import RetrievalResult


def retrieve(bundle: IndexBundle, query: str, *, top_k_text: int = 6, top_k_img: int = 4, merge_top_k: int = 8) -> List[RetrievalResult]:
    text_results = bundle.text.search(query, top_k=top_k_text) if top_k_text else []
    img_results = (
        bundle.image.search(query, top_k=top_k_img)
        if top_k_img and bundle.image is not None
        else []
    )
    table_results = (
        bundle.table.search(query, top_k=top_k_text)
        if top_k_text and bundle.table is not None
        else []
    )

    merged = text_results + table_results + img_results
    merged.sort(key=lambda r: r.score, reverse=True)
    return merged[:merge_top_k]
