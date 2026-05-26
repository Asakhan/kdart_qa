# K-DART-QA

다단계 수치 추론 한국어 문서 QA 데이터셋(K-DART-QA, 40문항) 구축 파이프라인. 한국 금융감독원 전자공시시스템(DART)의 사업보고서를 원천으로 한다.

본 코드는 *데이터셋 제작* 파이프라인이며, 검증(verification) 메커니즘이 LLM 에이전트의 정확도·비용에 미치는 영향을 다루는 후속 실험의 입력 데이터를 만든다.

## 파이프라인 개요

| Phase | 단계 | 산출물 |
|---|---|---|
| 1 | DART 수집 → 섹션 추출 → 청크 분할 → RAG 인덱스 | `data/raw`, `data/sections`, `data/chunks`, `data/index` |
| 2 | OpenAI(gpt-4o)로 40문항 초안 생성 (self-verification 포함) | `data/drafts/kdart_qa_draft_v1.json` |
| 2.5 | 검수 → 부족분 보충 생성·검수 반복 → 누적 40개 확보 | `data/drafts/*_supplement*.json`, `data/reviewed/*.json` |
| 3 | 다중 모델 calibration (Gemini 2.5/3.5/3.1-Lite, gpt-4o-mini) + 난이도 상향 변형 실험 | `data/calibration/*.json` |
| 4 | 최종 데이터셋 export (JSONL/Parquet + 통계) | `data/final/kdart_qa*.{jsonl,parquet,md}` |

## 사전 준비

### 환경 변수

프로젝트 루트의 `.env` 파일에 다음 키를 정의하라. 실행 시 `src/common.py`가
자동으로 로드한다. (export 해도 동일하게 동작함.)

```dotenv
DART_API_KEY=...     # https://opendart.fss.or.kr 발급
OPENAI_API_KEY=...   # 임베딩(text-embedding-3-small) + Phase 2 문항 초안 생성 + gpt-4o-mini calibration
GOOGLE_API_KEY=...   # Gemini Flash 계열 calibration
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
| OpenAI 임베딩 (text-embedding-3-small, 2,137 chunks) | ≈ 0.03 |
| OpenAI 문항 생성 (gpt-4o, 40문항 + 보충 + 재시도) | ≈ 1–2 |
| Calibration 4 모델 × 40문항 × 3시도 + 변형 실험 | ≈ 0.6–0.8 |
| **합계 (전 라운드 합)** | **≈ 2–3** |

각 단계 실행 직전에 예상 비용을 다시 표시하고 확인을 받는다.

## 실행 순서

각 단계는 독립 실행 가능하다. 결과 파일을 다음 단계가 읽는다.

```bash
# Phase 1
python scripts/01_fetch_dart.py          # DART 보고서 수집
python scripts/02_extract_sections.py    # 4개 섹션 추출
python scripts/03_build_index.py --yes --reset  # 청크 분할 + 임베딩 + ChromaDB 적재

# Phase 2
python scripts/04_generate_drafts.py     # 40문항 초안 생성

# Phase 2.5 — 사용자 검수
python scripts/05_review_session.py                          # 인터랙티브 검수
python scripts/05_review_session.py --filter T4              # 유형 필터
python scripts/05_review_session.py --review-mode second-pass  # 재검수

# 폐기로 인한 부족분 채우기 (필요 시 반복)
python scripts/04b_replenish_drafts.py --need "T1:1,T2:2,T3:6,T4:4,T5:3"

# Phase 3 — 모델 비교 calibration
python scripts/06_calibrate.py --yes \
    --model gemini-2.5-flash \
    --out data/calibration/results_gemini25flash.json
python scripts/06_calibrate.py --yes --model gemini-3.5-flash \
    --out data/calibration/results_gemini35flash.json
python scripts/06_calibrate.py --yes --model gemini-3.1-flash-lite \
    --out data/calibration/results_gemini31flashlite.json
python scripts/06b_calibrate_openai.py --yes \
    --out data/calibration/results_openai.json

# Phase 4 — 최종 export
python scripts/07_export.py              # Gemini-2.5 accepted-only 데이터셋
# (GPT 기준 quota 보존 통합본은 git history의 통합 스크립트 참고 →
#  결과는 data/final/kdart_qa_gpt_quota.jsonl)
```

각 스크립트는 `--help`로 옵션을 확인할 수 있다. 모든 단계가 `logs/` 아래 타임스탬프 로그를 남긴다.

## 실험에 사용할 데이터셋 파일

`data/final/` 아래에 세 가지 최종본이 있다.

| 파일 | 항목 수 | 특징 |
|---|---:|---|
| `kdart_qa.jsonl` | 4 | Gemini-2.5-Flash 기준 `accepted`만 선별 (가장 좁은 difficulty boundary) |
| `kdart_qa_gpt.jsonl` | 40 | gpt-4o-mini 기준 swap 적용. `task_type`은 변형의 실제 추론 패턴 반영 (T4 17개) |
| **`kdart_qa_gpt_quota.jsonl`** | **40** | **gpt-4o-mini 기준 swap + 원래 quota(T1:5/T2:10/T3:10/T4:10/T5:5) 보존 — 벤치마크용 권장** |

각 항목은 `calibration_*` 메타데이터(`classification`, `n_correct`, `n_attempts`, `retrieved_chunk_ids`)를 포함한다.

모델별 raw calibration 결과는 `data/calibration/results_{gemini25flash,gemini35flash,gemini31flashlite,openai}.json`에서 확인 가능 (4 모델 비교 연구용).

## 디렉토리 구조

```
kdart_qa/
├── README.md
├── requirements.txt
├── config.yaml
├── src/
│   ├── dart_client.py            # DART OpenAPI 클라이언트
│   ├── section_extractor.py      # 사업보고서 → 4개 섹션 추출 (lxml HTML 파서)
│   ├── table_converter.py        # HTML 표 → markdown 변환
│   ├── chunker.py                # 토큰 청크 분할
│   ├── rag_index.py              # OpenAI 임베딩 + ChromaDB 인덱스 (8000 토큰 truncation)
│   ├── question_generator.py     # OpenAI(gpt-4o) 문항 초안 생성 + self-verification
│   ├── review_helper.py          # Phase 2.5 검수 세션 상태 관리
│   ├── calibrator.py             # Provider-agnostic LLM 호출 (gemini | openai)
│   └── exporter.py               # 최종 export + 통계 마크다운
├── scripts/
│   ├── 01_fetch_dart.py
│   ├── 02_extract_sections.py
│   ├── 03_build_index.py
│   ├── 04_generate_drafts.py
│   ├── 04b_replenish_drafts.py   # 폐기로 부족분 발생 시 type별 타깃 재생성
│   ├── 05_review_session.py
│   ├── 06_calibrate.py           # --model / --out 플래그로 모델별 calibration
│   ├── 06b_calibrate_openai.py   # gpt-4o-mini 진입점
│   └── 07_export.py
├── data/
│   ├── raw/                      # (gitignore) DART 원본 ZIP
│   ├── sections/                 # (gitignore) 추출된 섹션 HTML/TXT
│   ├── chunks/                   # (gitignore) 토큰화된 JSONL
│   ├── index/                    # (gitignore) ChromaDB
│   ├── drafts/                   # 생성된 문항 초안 (main + supplements + variants)
│   ├── reviewed/                 # 사용자 검수 결과 (main + supplements)
│   ├── calibration/              # 4개 모델 × 다양한 라운드 calibration 결과
│   └── final/                    # 최종 JSONL/Parquet + statistics.md
├── tests/
└── logs/                          # (gitignore) 타임스탬프 로그
```

## 다중 모델 Calibration 결과 요약 (40문항 기준)

| 모델 | too_easy | accepted | too_hard | 비용 |
|---|---:|---:|---:|---:|
| Gemini-2.5-Flash | 26 | 4 | 10 | $0.232 |
| Gemini-3.5-Flash | 26 | 4 | 10 | $0.227 |
| Gemini-3.1-Flash-Lite | 26 | 3 | 11 | $0.214 |
| **gpt-4o-mini** | 26 | 0 | 14 | $0.074 |

- **Gemini 세대 간 90% 일치** — 모델 변경에 robust한 데이터셋
- **gpt-4o-mini가 multi-step에서 더 약함** (T4 too_hard 40% vs Gemini 10%)
- **9개 항목에 대해 공격적 변형 적용** 후 gpt-4o-mini 기준 accepted 3건 확보 (T2_004, T5_001, T4_017)

## 학술적 정직성

이 데이터셋은 학술 연구용으로, 다음 원칙을 따른다.

1. **Evidence-bounded**: 문항 생성 시 청크에 없는 외부 지식은 사용하지 않는다.
2. **추측 금지**: 청크에 근거가 없으면 폐기한다.
3. **로그 보존**: 모든 생성·검수 과정은 추적 가능한 로그로 보존된다 (`data/reviewed/changes.log` + `logs/`).
4. **편집 책임자 존중**: 사용자가 검수에서 내린 결정은 Phase 3에서 재최적화하지 않는다.
5. **Calibration 투명성**: 4개 모델 raw 결과 전체 commit되어 모델 비교·재현 가능.
