from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ..ingest.md_chunker import chunk_markdown
from .stores import IndexBundle


def build_index(
    md_path: str,
    images_dir: str | None,
    out_dir: str,
    *,
    caption_cache_path: str | None = None,
    chunk_size_chars: int = 1400,
    chunk_overlap_chars: int = 250,
) -> None:
    text_docs, image_docs, table_docs = chunk_markdown(
        md_path=md_path,
        images_dir=images_dir,
        caption_cache_path=caption_cache_path,
        chunk_size_chars=chunk_size_chars,
        chunk_overlap_chars=chunk_overlap_chars,
    )
    bundle = IndexBundle(out_dir)
    bundle.build_and_save(text_docs, image_docs, table_docs)

    print(f"Built index: {out_dir}")
    print(f"- text docs: {len(text_docs)}")
    print(f"- image docs: {len(image_docs)}")
    print(f"- table docs: {len(table_docs)}")


def _main() -> None:
    p = argparse.ArgumentParser(description="Build TF-IDF indices for DRAM spec (md + images).")
    p.add_argument("--md", required=False, help="Path to parsed spec markdown (.md)")
    p.add_argument("--images", required=False, help="Path to images directory (optional)")
    p.add_argument(
        "--caption_cache",
        required=False,
        help="Optional image caption cache (jsonl). See dram_rag/ingest/image_captioner.py",
    )
    p.add_argument("--out", required=False, default="./index_store", help="Output index directory")
    p.add_argument("--config", required=False, default=None, help="Path to settings.yaml (optional)")
    p.add_argument("--chunk_size_chars", type=int, default=1400)
    p.add_argument("--chunk_overlap_chars", type=int, default=250)

    args = p.parse_args()

    md_path = args.md
    images_dir = args.images
    out_dir = args.out
    caption_cache_path = args.caption_cache
    chunk_size_chars = args.chunk_size_chars
    chunk_overlap_chars = args.chunk_overlap_chars

    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        md_path = md_path or cfg.get("paths", {}).get("md_path")
        images_dir = images_dir or cfg.get("paths", {}).get("images_dir")
        out_dir = out_dir or cfg.get("paths", {}).get("index_dir")
        # caption cache는 config에 없을 수 있으므로, 있으면 사용
        caption_cache_path = caption_cache_path or cfg.get("paths", {}).get("caption_cache_path")
        chunk_size_chars = cfg.get("chunking", {}).get("chunk_size_chars", chunk_size_chars)
        chunk_overlap_chars = cfg.get("chunking", {}).get("chunk_overlap_chars", chunk_overlap_chars)

    if not md_path:
        raise SystemExit("--md is required (or set paths.md_path in --config).")

    build_index(
        md_path,
        images_dir,
        out_dir,
        caption_cache_path=caption_cache_path,
        chunk_size_chars=chunk_size_chars,
        chunk_overlap_chars=chunk_overlap_chars,
    )


if __name__ == "__main__":
    _main()
