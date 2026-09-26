# AI 사용 내역

## 사용한 도구

- **주 에이전트: Claude Code** (Anthropic, 원격 컨테이너 세션).
  - 코드·테스트·설정·문서 작성, 테스트와 스크립트 실행, 커밋을 수행했습니다.
- **하위 에이전트 2개** (같은 세션의 general-purpose agent).
  - WebSearch와 WebFetch로 arXiv 등에서 선행연구의 서지 정보와 본문을 조회했습니다. 파일은 쓰지 않았습니다.
  - 주 에이전트가 직접 다시 확인한 것은 8개 arXiv ID의 제목·날짜와 TTM GroupMatch 정의의 TeX 원문입니다.
  - 나머지 인용문과 수치는 하위 에이전트의 보고에 의존합니다. `docs/PRIOR_ART.md`에 본문 확인 여부를 표시했습니다.
- 유료 API, 모델 가중치 다운로드, 패키지 설치, GPU는 **사용하지 않았습니다.**

## AI가 결정한 것 (사람이 검토해야 함)

1. **설계 결정.**
   - 3-object 장면 고정(2-object는 누출 없는 분할이 불가능할 만큼 공간이 작음).
   - held-out pair 4개와 템플릿 3종.
   - caption-content 단위 누출 기준.
   - binding 그룹의 단어 다중집합 규칙: 객체 목록 구간만 강제하고 전체 캡션은 flag로 기록.
2. **head 구조와 손실 형태.** mean-pooled content, Hadamard binding, 그룹 간 cosine-InfoNCE edit-consistency.
3. **사전등록 값.** 최소관심효과 0.05, seed, 튜닝 grid, 자원 상한(`configs/prereg_pilot_v0.json`). 연구 책임자가 승인하거나 수정해야 합니다.
4. **blind 검출 허용치** `DETECT_TOL = 0.05`, 그리고 붕괴 검출 임계값(`bindcomp/collapse.py`).

## 알려진 AI 작업 중 오류와 수정

- **smoke 평가.** 수학적으로 같은 점수가 부동소수점 반올림 때문에 strict `>`에서 승리로 계산되었습니다. 상대 허용치 1e-9의 tie 처리를 추가했습니다(`evaluate.gt`).
- **테스트 fixture.** 한 fixture가 Winoground 정의에 대해 잘못된 기대값을 담고 있어 fixture를 고쳤습니다. 구현은 바꾸지 않았습니다.
- **mutation test 중 bytecode 캐시.** 오래된 `.pyc` 때문에 원복 뒤에도 실패가 나왔습니다. `__pycache__`를 삭제하고 재실행했습니다.
- **분할 artifact 2건.** 발견 즉시 수정하고 `STATUS.md`에 기록했습니다.

## 사람이 확인해야 할 것

- `docs/PRIOR_ART.md`의 PARTIAL / FULL_TEXT_UNVERIFIED 항목, 그리고 신규성 판단.
- 수동 gradient 식. finite-difference gradcheck와 mutation test로 확인했지만, 수치 검사는 증명이 아닙니다.
- proxy feature(oracle object token) 가정이 연구 질문에 적절한지.

## 픽셀 입력 단계 (2026-09-26)

- **수행 범위.** 같은 Claude Code 세션이 다음을 했습니다. 하위 에이전트는 쓰지 않았습니다.
  - 입력 출처 추적.
  - renderer, CLIP feature 경로, torch bridge, data_v1, CPU ledger 구현.
  - `.venv` 설치, 단일 encoder 다운로드, stage 실행, 문서 갱신.
- **외부 접근.** PyPI와 download.pytorch.org에서 패키지를, Hugging Face에서 checkpoint 1개를 받았습니다. 그 전에 모델 카드와 PyPI 메타데이터의 사용권을 직접 읽었습니다. 원격 코드 실행과 유료 API는 사용하지 않았습니다.
- **AI가 정한 것 (검토 필요).**
  - renderer의 시각 표현(특히 metal/rubber).
  - token 선택: CLIP 최종층 class token + patch token, 표준화 bridge.
  - panel 규칙, 학습 설정(lr 0.01, 300 step, seed 0–2), 판정 임계값(fit ≥ 0.75, 손실 비 ≤ 0.7).
  - orbit 분할 비율.
- **AI 작업 중 발견하고 수정한 오류.**
  - 원뿔 반사광 위치(렌더 의미 손실).
  - `/usr/bin/time` 부재로 설치 첫 시도 실패.
  - 커밋 전 작업 트리에서 stage를 실행한 일(재실행 결과 동일).
  - manifest의 dirty 오탐 버그.
