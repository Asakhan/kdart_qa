# K-DART-QA

다단계 수치 추론 한국어 문서 QA 데이터셋(K-DART-QA, 40문항) 구축 파이프라인. 한국 금융감독원 전자공시시스템(DART)의 사업보고서를 원천으로 한다.

본 코드는 *데이터셋 제작* 파이프라인이며, 검증(verification) 메커니즘이 LLM 에이전트의 정확도·비용에 미치는 영향을 다루는 후속 실험의 입력 데이터를 만든다.

## 파이프라인 개요

| Phase | 단계 | 산출물 |
|---|---|---|
| 1 | DART 수집 → 섹션 추출 → 청크 분할 → RAG 인덱스 | `data/raw`, `data/sections`, `data/chunks`, `data/index` |
| 2 | OpenAI로 40문항 초안 생성 (self-verification 포함) | `data/drafts/kdart_qa_draft_v1.json` |
| 2.5 | 사용자(편집 책임자) 검수 (CLI 도구) | `data/reviewed/kdart_qa_reviewed_v1.json` |
| 3 | gemini-2.5-flash 난이도 calibration → 최종 출력 | `data/calibration/results.json`, `data/final/kdart_qa.{jsonl,parquet}` |

## 사전 준비

### 환경 변수

프로젝트 루트의 `.env` 파일에 다음 키를 정의하라. 실행 시 `src/common.py`가
자동으로 로드한다. (export 해도 동일하게 동작함.)

```dotenv
DART_API_KEY=...     # https://opendart.fss.or.kr 발급
OPENAI_API_KEY=...   # 임베딩(text-embedding-3-small) + Phase 2 문항 초안 생성
GOOGLE_API_KEY=...   # Phase 3 calibration (gemini-2.5-flash)
```

`.env`는 `.gitignore`에 포함되어 있으므로 커밋되지 않는다.

### 파이썬 환경

Python 3.11 권장(3.12에서도 동작). 가상환경 사용을 권장한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 예상 비용

| 항목 | 금액 (USD) |
|---|---|
| OpenAI 임베딩 (text-embedding-3-small) | ≈ 1–2 |
| OpenAI 문항 생성 (gpt-4o, 40문항 + 재시도) | ≈ 2–5 |
| Gemini Flash calibration (40문항 × 3시도) | ≈ 5–10 |
| **합계** | **≈ 8–17** |

각 단계 실행 직전에 예상 비용을 다시 표시하고 확인을 받는다.

## 실행 순서

각 단계는 독립 실행 가능하다. 결과 파일을 다음 단계가 읽는다.

```bash
# Phase 1
python scripts/01_fetch_dart.py          # DART 보고서 수집
python scripts/02_extract_sections.py    # 4개 섹션 추출
python scripts/03_build_index.py         # 청크 분할 + 임베딩 + ChromaDB 적재

# Phase 2
python scripts/04_generate_drafts.py     # 40문항 초안 생성

# Phase 2.5 — 사용자 검수
python scripts/05_review_session.py                          # 1차 검수
python scripts/05_review_session.py --filter T4              # 유형 필터
python scripts/05_review_session.py --review-mode second-pass  # 재검수

# Phase 3
python scripts/06_calibrate.py           # gemini로 난이도 calibration
python scripts/07_export.py              # 최종 데이터셋 + 통계
```

각 스크립트는 `--help`로 옵션을 확인할 수 있다. 모든 단계가 `logs/` 아래 타임스탬프 로그를 남긴다.

## 디렉토리 구조

```
kdart_qa/
├── README.md
├── requirements.txt
├── config.yaml
├── src/
│   ├── dart_client.py
│   ├── section_extractor.py
│   ├── table_converter.py
│   ├── chunker.py
│   ├── rag_index.py
│   ├── question_generator.py
│   ├── review_helper.py
│   ├── calibrator.py
│   └── exporter.py
├── scripts/
│   ├── 01_fetch_dart.py
│   ├── 02_extract_sections.py
│   ├── 03_build_index.py
│   ├── 04_generate_drafts.py
│   ├── 05_review_session.py
│   ├── 06_calibrate.py
│   └── 07_export.py
├── data/{raw,sections,chunks,index,drafts,reviewed,calibration,final}/
├── tests/
└── logs/
```

## 학술적 정직성

이 데이터셋은 학술 연구용으로, 다음 원칙을 따른다.

1. **Evidence-bounded**: 문항 생성 시 청크에 없는 외부 지식은 사용하지 않는다.
2. **추측 금지**: 청크에 근거가 없으면 폐기한다.
3. **로그 보존**: 모든 생성·검수 과정은 추적 가능한 로그로 보존된다.
4. **편집 책임자 존중**: 사용자가 검수에서 내린 결정은 Phase 3에서 재최적화하지 않는다.
