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
OPENAI_API_KEY=...   # Phase 2 문항 초안 생성(gpt-4o) + gpt-4o-mini calibration (임베딩에는 불필요)
GOOGLE_API_KEY=...   # Gemini Flash 계열 calibration
```

`.env`는 `.gitignore`에 포함되어 있으므로 커밋되지 않는다.

> **임베딩은 무료 로컬 모델**(기본 `BAAI/bge-m3`, `rag.provider: local`)을 쓰므로
> 인덱스 빌드·검색·calibration의 임베딩 단계는 **`OPENAI_API_KEY` 없이** 동작하고
> 비용이 0이다. `OPENAI_API_KEY`는 문항 생성(gpt-4o)과 gpt-4o-mini calibration에서만
> 필요하다. `config.yaml`의 `rag.provider`를 `openai`로 바꾸면 기존 OpenAI 임베딩으로
> 되돌릴 수 있다(이 경우 `rag.model`/`embedding_dim`도 함께 변경 후 `--reset` 재빌드).

### 파이썬 환경

Python 3.11 권장(3.12에서도 동작). 가상환경 사용을 권장한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# GPU가 없으면 CPU 전용 torch가 더 작고 빠르다:
#   pip install --index-url https://download.pytorch.org/whl/cpu torch
```

로컬 임베딩은 `sentence-transformers`(`BAAI/bge-m3`, 1024차원)를 쓴다. GPU가 있으면
자동 사용하고 없으면 CPU로 폴백한다(`rag.device: auto`). 첫 실행 시 모델 가중치
(~2.3GB)를 HuggingFace에서 1회 내려받는다.

> 참고: `sentence-transformers`가 끌어오는 `tokenizers`가 `chromadb`의 상한 핀과
> 충돌한다는 pip 경고가 뜰 수 있으나, 임베딩에 우리 자체 함수를 쓰므로 런타임에는
> 영향이 없다.

저사양 CPU(소량 RAM)에서는 `rag.max_batch_tokens`(기본 6144)가 배치 메모리를
제한해 bge-m3(fp32 ~2.3GB)가 OOM 없이 동작하도록 한다.

### 예상 비용

| 항목 | 금액 (USD) |
|---|---|
| 임베딩 (로컬 `BAAI/bge-m3`, 2,137 chunks) | **0 (무료, 로컬 실행)** |
| OpenAI 문항 생성 (gpt-4o, 40문항 + 보충 + 재시도) | ≈ 1–2 |
| Calibration 4 모델 × 40문항 × 3시도 + 변형 실험 | ≈ 0.6–0.8 |
| **합계 (전 라운드 합)** | **≈ 1.6–2.8** |

문항 생성·calibration의 LLM 호출 직전에는 예상 비용을 다시 표시하고 확인을 받는다.
임베딩은 무료라 비용 확인 프롬프트가 없다.

## 실행 순서

각 단계는 독립 실행 가능하다. 결과 파일을 다음 단계가 읽는다.

```bash
# Phase 1
python scripts/01_fetch_dart.py          # DART 보고서 수집
python scripts/02_extract_sections.py    # 4개 섹션 추출
python scripts/03_build_index.py --yes --reset  # 청크 분할 + 로컬 임베딩 + ChromaDB 적재
                                                # (--reset로만 강제 재빌드, 평소엔 재사용)
python scripts/08_eval_retrieval.py      # 검색 recall 측정 (목표 ≥ 0.90)
python scripts/08_eval_retrieval.py --show-misses --top-k 15  # 미스 문항 확인 / k 스윕

# Phase 2
#
# 로컬 bge-m3 인덱스 기준 측정 결과(kdart_qa_gpt_quota.jsonl, company+year 필터):
#   recall@8=0.75, @15=0.85, @20=0.925.  → rag.top_k=20에서 목표 0.90 달성.
#   미달의 원인은 필터/임베딩이 아니라 순위 깊이: 신한지주·KB금융처럼 청크 풀이
#   175~318개인 대형 보고서에서 gold가 깊은 순위(최대 58위)에 잡힌다. gold는 항상
#   풀 안에 있으므로 k를 늘리면 해결된다(동일 k를 FinQA 비교에도 적용).
python scripts/04_generate_drafts.py     # 40문항 초안 생성

# Phase 2.5 — 사용자 검수
python scripts/05_review_session.py                          # 인터랙티브 검수
python scripts/05_review_session.py --filter T4              # 유형 필터
python scripts/05_review_session.py --review-mode second-pass  # 재검수

# 폐기로 인한 부족분 채우기 (필요 시 반복)
python scripts/04b_replenish_drafts.py --need "T1:1,T2:2,T3:6,T4:4,T5:3"

# Phase 3 — 모델 비교 calibration (문항 간 병렬 실행, --max-workers로 조절)
python scripts/06_calibrate.py --yes \
    --model gemini-2.5-flash --max-workers 4 \
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

# Phase 4 — 비교 실험용 파생 산출물 (전부 오프라인, API/인덱스 불필요)
python scripts/10_export_finqa_matched.py   # 난이도 정렬 서브셋 (hops ≤ 2)
python scripts/09_export_finqa_format.py    # FinQA 스키마 단일 JSON 배열 (근거 동봉)
python scripts/09_export_finqa_format.py \
    --input data/final/kdart_qa_finqa_matched.jsonl   # 매칭 서브셋의 FinQA 포맷
```

calibration은 문항 간 `ThreadPoolExecutor` 병렬화로 LLM 왕복을 겹쳐 40문항 소요
시간을 크게 줄인다. worker 수는 `calibration.max_workers`(기본 4) 또는
`--max-workers`로 제어하며, `1`로 두면 기존 순차 동작과 동일하다. 증분 저장은
완료 콜백이 메인 스레드에서 순차 실행되어 thread-safe하다.

각 스크립트는 `--help`로 옵션을 확인할 수 있다. 모든 단계가 `logs/` 아래 타임스탬프 로그를 남긴다.

## 실험에 사용할 데이터셋 파일

`data/final/` 아래 최종본:

| 파일 | 항목 수 | 특징 |
|---|---:|---|
| `kdart_qa.jsonl` | 4 | Gemini-2.5-Flash 기준 `accepted`만 선별 (가장 좁은 difficulty boundary) |
| `kdart_qa_gpt.jsonl` | 40 | gpt-4o-mini 기준 swap 적용. `task_type`은 변형의 실제 추론 패턴 반영 (T4 17개) |
| **`kdart_qa_gpt_quota.jsonl`** | **40** | **gpt-4o-mini 기준 swap + 원래 quota(T1:5/T2:10/T3:10/T4:10/T5:5) 보존 — 벤치마크용 권장** |
| `kdart_qa_finqa_matched.jsonl` | 28 | **언어 효과 분리용** — `reasoning_hops ≤ 2` 서브셋. FinQA의 1–2 step 다수 분포에 추론 난이도를 정렬한 비교군. 각 항목에 `difficulty_group`(`1-2_step`/`3_step`/`4plus_step`)·`finqa_matched` 라벨 추가. 원본 40문항·검수 결정은 그대로 보존. |
| `kdart_qa_finqa_format.json` | 40 | **FinQA와 동일 스키마**(`pre_text`/`post_text`/`table`/`qa{...}`)의 단일 JSON 배열. **근거 본문이 항목에 동봉**되어 검색 불필요. `program`은 빈 문자열, 채점은 `exe_ans`(파싱된 수치) 기반(program-free). 원본 K-DART 메타는 `kdart_meta`에 보존. 실험에는 이 파일 하나만 제공하면 된다. |
| `kdart_qa_finqa_matched_finqa_format.json` | 28 | `kdart_qa_finqa_matched.jsonl`을 FinQA 스키마로 변환한 것(난이도 정렬 + FinQA 포맷). |

K-DART 원본 형식 항목은 `calibration_*` 메타데이터(`classification`, `n_correct`, `n_attempts`, `retrieved_chunk_ids`)를 포함한다.

#### FinQA 스키마 출력의 용도·한계

- `kdart_qa_finqa_format.json`은 FinQA 원본과 K-DART를 **하나의 스키마로 동시에**
  읽도록 만든 변환본이다. 각 항목에 근거(표/산문)가 동봉되어 ChromaDB·청크 파일·
  임베딩·LLM 없이 채점할 수 있다.
- `program` 필드는 **비어 있다**(추론 프로그램을 합성하지 않음 — 설계 결정).
  따라서 FinQA 평가 코드가 program 실행으로 채점하는 모드 대신 **`answer`/`exe_ans`
  기반 채점**(program-free)을 사용해야 한다.
- T2~T4의 다단계 계산 문항은 FinQA와 마찬가지로 **정답이 근거에 직접 나타나지 않고
  계산으로 도출**된다(근거에는 피연산자가 들어 있고 `exe_ans`가 계산 결과).
- 변환·검증은 `scripts/09_export_finqa_format.py`로 언제든 오프라인 재생성 가능
  (`data/final/*.jsonl` + `data/chunks/*.jsonl`만 사용).

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
│   ├── rag_index.py              # 로컬(sentence-transformers) 임베딩 + ChromaDB 인덱스
│   │                             #   (provider 스위치로 OpenAI 임베딩 복귀 가능)
│   ├── finqa_adapter.py          # K-DART → FinQA 스키마 변환 (근거 동봉, 오프라인)
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
│   ├── 06_calibrate.py           # --model / --out / --max-workers 로 모델별 병렬 calibration
│   ├── 06b_calibrate_openai.py   # gpt-4o-mini 진입점 (--max-workers 지원)
│   ├── 07_export.py
│   ├── 08_eval_retrieval.py      # 검색 recall 측정 (gold가 top-k에 포함되는 비율)
│   ├── 09_export_finqa_format.py # FinQA 스키마 단일 JSON 배열 export (근거 동봉)
│   └── 10_export_finqa_matched.py # 난이도 정렬(hops ≤ 2) 서브셋 export
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
