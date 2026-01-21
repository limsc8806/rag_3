# DRAM Adaptive RAG Agent (MD + Images)

이 프로젝트는 **DRAM spec PDF를 미리 파싱해둔 결과물(.md + 이미지 폴더)** 을 입력으로 받아,
- 텍스트 청크 인덱스 (TF-IDF)
- 이미지(캡션-라이트) 인덱스 (TF-IDF)
를 구축하고, **Adaptive RAG 흐름(검색 → 관련성 평가 → 생성 → 답변 평가 → 쿼리 재작성 루프)** 로 질의응답을 수행하는 RAG 에이전트의 **MVP 구현**입니다.

본 코드는 첨부 실습 노트북의 구조(Adaptive RAG 컨트롤 플로우)를 유지하되,
LangChain/LangGraph 없이도 동작하도록 **표준 Python + scikit-learn** 기반으로 구성했습니다.

- 생성(LLM) 호출은 옵션입니다.
- 기본값은 오프라인에서도 동작 가능한 **추출형(extractive) 답변** 모드입니다.

## 1) 입력 형태

- `spec.md` : PDF를 파싱해 얻은 단일 마크다운 파일
- `images/` : 마크다운 내 링크된 이미지 파일들 (예: `![](images/fig1.png)`)

## 2) 설치

```bash
cd dram_adaptive_rag_agent
pip install -r requirements.txt
```

> OpenAI 모델을 사용할 계획이 없으면 `openai`는 설치하지 않아도 됩니다.

## 3) 설정

`config/settings.yaml`에서 아래 값을 채우거나, CLI 인자로 주입할 수 있습니다.

- `paths.md_path`: spec.md 경로
- `paths.images_dir`: images 폴더 경로(선택)
- `paths.index_dir`: 인덱스 저장 경로
- `paths.caption_cache_path`: (선택) 이미지 캡션 캐시(jsonl) 경로

## 4) 인덱스 생성

### (A) CLI 인자로 생성

```bash
python -m dram_rag.index.build_index \
  --md /path/to/spec.md \
  --images /path/to/images \
  # 옵션: 이미지 캡션 캐시(jsonl)
  --caption_cache /path/to/captions.jsonl \
  --out ./index_store
```

### (B) config 기반 생성

```bash
python -m dram_rag.index.build_index --config ./config/settings.yaml
```

## 5) 채팅 실행

```bash
python -m dram_rag.app.chat --config ./config/settings.yaml
```

## 6) LLM(옵션) 연결

### OpenAI 사용

1) 패키지 설치

```bash
pip install openai
```

2) 환경 변수

```bash
export OPENAI_API_KEY="..."
```

3) `config/settings.yaml` 수정

- `llm.provider: "openai"`
- `llm.model: "..."` (사용 환경에 맞게)

## 7) 설계 메모

- 텍스트와 이미지를 **별도 인덱스**로 저장 후, 검색 단계에서 merge합니다.
- 이미지 인덱스는 현재 "캡션-라이트"(ALT 텍스트 + 주변 문맥) 방식입니다.
  - 추후 Vision LLM 캡셔닝을 붙이면 이미지 검색 품질이 크게 개선됩니다.
- 실습 노트북의 LLM grader(문서 관련성 평가/환각 평가)는 현재 TF-IDF 유사도 및 간단한 규칙 기반으로 대체했습니다.
  - 운영 환경에서는 grader를 LLM으로 교체하는 것을 권장합니다.

## 8) 폴더 구조

```text
dram_rag/
  ingest/         # md 로드/섹션 분리/청킹, 이미지 참조 추출
  index/          # TF-IDF 인덱스 빌드/저장/로드
  retrieval/      # 검색/merge, 관련성 필터, 쿼리 재작성
  generation/     # 프롬프트/LLM 클라이언트/추출형 답변
  agent/          # Adaptive RAG 컨트롤 플로우
  app/            # CLI
```