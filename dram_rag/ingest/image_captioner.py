from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class CaptionCache:
    """Simple caption cache.

    File format (jsonl):
      {"image_resolved_path": "/abs/path/to.png", "caption": "..."}
    """

    captions: Dict[str, str]

    @classmethod
    def load(cls, path: str | Path) -> "CaptionCache":
        path = Path(path)
        captions: Dict[str, str] = {}
        if not path.exists():
            return cls(captions)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                p = obj.get("image_resolved_path")
                c = obj.get("caption")
                if p and c:
                    captions[str(p)] = str(c)
        return cls(captions)

    def get(self, image_resolved_path: str) -> Optional[str]:
        return self.captions.get(str(image_resolved_path))


# NOTE:
# Vision LLM 캡셔닝을 붙이려면, 별도 스크립트에서 이미지별 caption을 생성하고
# 위 jsonl 포맷으로 저장한 뒤 build_index 시에 주입하는 방식을 권장합니다.
