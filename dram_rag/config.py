from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from .generation.llm_clients import LLMConfig


class PathsConfig(BaseModel):
    md_path: str = ""
    images_dir: str = ""
    index_dir: str = "./index_store"
    caption_cache_path: str = ""


class ChunkingConfig(BaseModel):
    chunk_size_chars: int = 1400
    chunk_overlap_chars: int = 250


class RetrievalConfig(BaseModel):
    top_k_text: int = 6
    top_k_img: int = 4
    merge_top_k: int = 8
    min_relevance_score: float = 0.12


class AgentConfig(BaseModel):
    max_loops: int = 2


class Settings(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_settings(path: str | Path) -> Settings:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Settings.model_validate(cfg)
