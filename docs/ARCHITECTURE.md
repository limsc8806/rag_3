# DRAM Adaptive RAG Agent - 아키텍처 노트

이 문서는 런타임 흐름과 모듈 연결 구조를 설명합니다. 핵심 로직이나
데이터 포맷이 바뀌면 함께 갱신하세요.

## 전체 흐름

1) **Ingest**
   - `dram_rag/ingest/md_loader.py`: 마크다운 읽기 및 헤딩 기준 분리.
   - `dram_rag/ingest/image_extractor.py`: 이미지 참조 및 주변 문맥 추출.
   - `dram_rag/ingest/md_chunker.py`: 아래 문서 타입 생성:
     - 텍스트 청크 (`doc_type="text"`)
     - 이미지 문서(캡션-라이트) (`doc_type="image"`)
     - HTML `<table>` 또는 마크다운 파이프 테이블 문서 (`doc_type="table"`)

2) **Index Build**
   - `dram_rag/index/build_index.py`: 인덱싱 파이프라인 실행.
   - `dram_rag/index/stores.py`:
     - `TfidfIndex`: 벡터라이저/매트릭스/문서 JSONL 저장.
     - `IndexBundle`: `text`, `image`, `table` 인덱스 묶음.
   - 출력 폴더(기본 `./index_store`):
     - `text.docs.jsonl`, `image.docs.jsonl`, `table.docs.jsonl`
     - `*.vectorizer.joblib`, `*.matrix.npz`
     - `index_metadata.json` (빌드 시 scikit-learn 버전 기록)

3) **Retrieval**
   - `dram_rag/retrieval/retrievers.py`: text/table/image 검색 후 병합.
   - `dram_rag/retrieval/graders.py`: 관련성 필터(TF-IDF 임계치).
   - `dram_rag/retrieval/query_rewrite.py`: 결정적 쿼리 리라이트 루프.

4) **Generation**
   - `dram_rag/generation/rag_chain.py`: 추출형 답변 또는 LLM 답변.
   - `dram_rag/generation/llm_clients.py`: OpenAI 또는 no-op.
   - `dram_rag/generation/answer_grader.py`: 휴리스틱 답변 평가.
   - `dram_rag/generation/prompts.py`: 시스템 프롬프트/컨텍스트 포맷.

5) **Agent + App**
   - `dram_rag/agent/graph.py`: Adaptive RAG 제어 루프.
   - `dram_rag/app/chat.py`: CLI 인터페이스.

## 설정

- `config/settings.yaml`: 경로, 청킹, 리트리벌, 루프, LLM 설정.
- `dram_rag/config.py`: 설정 로드 및 검증.

## 업데이트 규칙

아래 영역을 바꾸면 이 문서를 함께 갱신하세요.

- 인제스트 로직: `dram_rag/ingest/*`
- 인덱스 포맷/스키마: `dram_rag/index/*`
- 리트리벌/스코어링 동작: `dram_rag/retrieval/*`
- 에이전트 제어 흐름: `dram_rag/agent/*`
- 생성/프롬프트: `dram_rag/generation/*`

리트리벌 동작 또는 문서 스키마가 바뀌면 다음도 점검합니다.

- `tests/regression_questions.yaml`
- `tests/test_regression.py`

## 보완 필요 사항 / 개선 아이디어

- **인덱스 호환성**: scikit-learn 버전 불일치 시 joblib 경고가 발생합니다.
  인덱스 생성/실행 버전을 맞추는 것을 권장합니다.
  `index_metadata.json`에 저장된 버전과 런타임 버전이 다르면 경고합니다.
- **테이블 파싱**: HTML/마크다운 파이프 테이블을 처리합니다.
- **이미지 캡션**: 현재 캡션-라이트(ALT/문맥) 기반입니다.
  비전 캡셔닝을 추가하면 이미지 검색 품질이 개선됩니다.
- **랭킹**: text/table/image를 동일 가중으로 병합합니다.
  모달리티 가중치 또는 table 전용 top-k를 고려하세요.
- **쿼리 리라이트**: 규칙 기반만 사용합니다.
  LLM/언어 지식 기반 리라이트가 필요합니다.
- **답변 평가**: 휴리스틱입니다.
  운영 환경에서는 LLM grader 도입이 적합합니다.
- **E2E 테스트**: 현재는 리트리벌만 검증합니다.
  답변 품질을 검증하는 E2E 테스트가 필요합니다.
