# STATUS: P3 Learning Object-Attribute Binding that Composes Beyond Training Pairs

마지막 갱신: 2026-09-26 (paired 비교 `paired_v1` + 원고 Working Draft v1).
- 인수 기준: `1edc535`. 최종 tested code: `0b0d130`.
- 이번 단계 코드: `fc7d5f6`(paired 실행 커밋). 원고 v0: `6d6887e`.
- 이전 기록(A절, 1단계 기록)은 아래에 원문 그대로 보존했다.

# B. Paired 비교와 원고 (2026-09-26)

## B0. 판정

| 구분 | 판정 | 근거 |
|---|---|---|
| 소프트웨어 | **TECHNICAL_TEST_PASS** | `.venv`에서 unittest 70/70 통과, skip 0(`runs/tests_20260926T231246Z`) |
| split 의미 검사 | **완료 (metadata 전용)** | edit orbit, 수락률, orbit filter 유지율, 구성비. test 점수는 열지 않음 |
| paired 비교 | **OPTIMIZATION_OR_INPUT_UNRESOLVED** (사전 고정 규칙) | 두 arm 모두 fit 기준 미달(seed 평균 BN train group A 0.251, B 0.514 < 0.75). 학습이 두 arm 모두에서 불안정 |
| edit-loss 분기 | **ON_HOLD** | 규칙상 PRELIMINARY_ADDED_UTILITY가 아니면 보류. 결과를 보고 자동으로 후속 실행하지 않음 |
| 과학 | **SCIENCE_NOT_EVALUATED** (dev 전용) | test는 봉인 상태. 합성 RGB와 frozen encoder에 한정 |
| 원고 | **WORKING_DRAFT_V1_BUILT**, HUMAN_REVIEW_PENDING | `paper/main.pdf` 13쪽. 본문은 8쪽에서 끝남(한도 9). 공식 ICLR 2027 style이며 미제출 |

## B1. 실제 명령과 결과 위치

| 명령 | 결과 위치 | CPU |
|---|---|---|
| `python3 scripts/ledger_run.py --name split_semantics_v1_committed --ledger runs/paired_v1_cpu_ledger.json --cap-s 7200 -- python3 scripts/analyze_split_semantics.py` | `runs/split_semantics_v1_20260926T230848Z`. 먼저 dirty tree에서 한 실행 `…225449Z`도 보존했고, 결과는 float 한 자리 차이를 빼면 동일 | 10.7 s |
| `.venv/bin/python scripts/paired_v1.py` (커밋 fc7d5f6, clean) | `runs/paired_v1_20260926T230018Z` (`paired_summary.json`, `arms_detail.json`, head `.pt` 6개, log, manifest) | 471.3 s, peak RSS 1.94 GiB |
| debug(8/8 panel, 5 step) | `runs/debug_paired_tiny_20260926T225836Z` | 27.3 s |
| `python3 scripts/ledger_run.py --name final_tests_paired_stage ... -- .venv/bin/python scripts/run_tests.py` | `runs/tests_20260926T231246Z` | 33.1 s |
| `python3 scripts/ledger_run.py --name paper_build ... --ledger runs/paper_build_cpu_ledger.json --cap-s 600 -- bash paper/build.sh` | `paper/main.pdf` | 빌드 총 13.3 s / 600 |

- **task ledger** (`runs/paired_v1_cpu_ledger.json`): 562.5 / 7,200 s. 개발 회귀 테스트 8.9 s 포함.
- **설치(상한 밖에 별도 기록)**
  - TeX Live 2023: `runs/install_latex_*`, 35 CPU s.
  - poppler-utils: `runs/install_poppler_*`. 첫 시도는 오래된 index 때문에 실패했고, 그 기록도 보존.
  - 공식 ICLR 2027 style ZIP(sha256 `0d940dfa…`). 새 연구 데이터나 가중치는 받지 않았다.

## B2. Split 의미 (지시 1)

- **edit orbit의 정의**: binding 편집(한 속성 종류를 두 객체 사이에서 교환)과 relation 편집(slot 교환)이 생성하는 군의 orbit이다. 중간 장면의 유효성을 무시하면, shape·color·material multiset이 같은 유효 장면의 집합과 같다. content 편집(속성·객체 교체)은 다른 orbit으로 옮긴다.
- **orbit filter**: binding·relation 편집은 100% 유지된다. content 편집의 유지율은 train 50.1%, dev 10.4%, calib 4.7%, test 템플릿 3.4%다. 따라서 **수락된 content 편집은 균등 표본이 아니다.**
- **구성비 이동**: 작은 split은 orbit 수가 적어 구성비가 train과 다르다(dev 색 분포 TV 0.12).
- **BN 비율**: 단일 편집 split은 50%, test composition은 30%다.
- **Panel**: train 512 그룹(427 orbit), dev 128 그룹(74 orbit). 공유 orbit 0이며, dev는 이전에 채점된 32 그룹을 제외했다.

## B3. Paired 결과 (dev 128 그룹, BN 64; 가설 검정 아님)

| seed 0/1/2 | A: hard-neg | B: +edit (λ=0.01) | B−A |
|---|---|---|---|
| dev BN group | .094/.016/.000 | .000/.125/.234 | −.094/.109/.234 |
| dev BN GroupMatch | .688/.641/.656 | .609/.812/.719 | −.078/.172/.062 |
| train BN group (fit) | .535/.152/.066 | .094/.746/.703 | |
| 최종 task loss | 0.51/1.83/7.18 | 0.91/0.25/0.29 | |

- **판정**: 두 arm 모두 fit 기준 미달이므로 **OPTIMIZATION_OR_INPUT_UNRESOLVED**.
- **학습 불안정**: 로그된 grad norm의 run별 최대값은 77.3–427.7이다. A seed 2는 최종 loss 7.18로, 초기값 3.56보다 높다.
- **B−A 차이**: 평균 0.083, group bootstrap CI [0.036, 0.135]. 이 CI는 seed 분산을 반영하지 않는다. B가 이긴 seed는 A가 높은 loss로 끝난 run과 겹치므로 **추가 효용으로 해석하지 않는다.**
- **chance와의 관계**: 두 arm 모두 BN group score는 1/6 아래(A 0.036, B 0.120)이고 GroupMatch는 1/2 위(A 0.661, B 0.714)다.
- **zero-shot CLIP** (BN): group 0.031, GroupMatch 0.594.
- **λ 공개**: λ=0.01은 기존 grid에서 가장 작은 0이 아닌 값이고, 그 grid는 gradient 관찰 이후에 정해졌다. init 시점 edit/task gradient 비는 13.1이고, λ를 곱하면 약 0.13이다.

## B4. 미실행 (NOT_RUN)
- 안정화된 재비교(낮은 lr, schedule, clipping, 정규화 score, 평균화). 새 사전 고정 config와 승인이 필요하다.
- v1 test 평가, 자연 이미지 전이, encoder 교체·층 선택·fine-tuning, calibrated pair accuracy, 12-setting sweep.

## B5. 원고
- 파일: `paper/main.tex`, `paper/sections/*.tex`, `paper/appendix.tex`, `paper/references.bib`, `paper/tables/*.tex`(생성), `paper/figures/dev_examples_row.png`(run 산출물 재배치), `paper/claim_evidence.tsv`(35건), `paper/BUILD.md`, `paper/PAPER_STATUS.md`, `paper/main.pdf`.
- 숫자: `scripts/export_paper_numbers.py`가 원자료에서 매크로 327개와 표 4개를 만든다. 출처는 `paper/generated/provenance.tsv`에 있다.
- 남은 주의사항: 사람 검토 없음. arXiv export의 연도가 최신 버전 기준이라 인용 연도 정책이 필요하다. C29의 설명은 미검정이다.

---

# A. 픽셀 입력 단계 (pixel_baseline_v1)

(당시 머리말, 원문 보존)
마지막 갱신: 2026-09-26 (픽셀 입력 단계 `pixel_baseline_v1`).
- 코드 기준 커밋: `121fdab`(stage 코드). `0b0d130`은 manifest 버그 수정이다.
- 인수 기준 커밋은 `f606795`이며, 그 이전 기록은 아래 "1단계 기록"에 원문 그대로 보존했다.

## A0. 판정

| 질문 | 판정 | 근거 (사전 고정 규칙: `configs/pixel_baseline_v1.json` `decision_rules`) |
|---|---|---|
| 소프트웨어 | **TECHNICAL_TEST_PASS** | `.venv`에서 unittest 68/68 통과, skip 0(`runs/tests_20260926T162256Z`, 소스 hash가 커밋 `0b0d130`과 일치). data_v1 감사 FAIL 0. |
| Q1. 픽셀만으로 모델 입력을 만들고 평가 경로가 동작하는가 | **PASS** | renderer decode 불일치 0/288. feature는 모두 유한하고 분산 > 0. oracle 대조 1.0. 계획한 cell이 모두 있음. provenance-guard 테스트 통과. |
| Q2. hard-negative 기준선이 학습되는가 | **PASS** | fit sanity: 학습용 16 그룹에서 group 1.000(규칙 ≥ 0.75). 마지막/첫 step 손실 비의 seed 평균 0.14(규칙 ≤ 0.7). |
| Q2′. 병목은 어디인가 (기술 판독, 유의성 없음) | 평가 경로는 **아님**. dev의 binding-necessary 사례로 **일반화되는 것은 관찰되지 않음**. encoder, 데이터 크기(64 그룹), head 중 무엇이 원인인지는 **분리하지 못함**. | A4절 |
| Q3. 정답을 입력에 주지 않고 binding-necessary 사례를 평가할 수 있는가 | **PASS** | dev의 binding-necessary 16 그룹을 픽셀과 캡션만으로 점수화했다. 그 위에서 oracle은 1.0이다. shuffle·blind·content-only 대조를 pair AUC, tie 비율, 우연 수준과 함께 보고했다. |
| 종합 | **PIXEL_BASELINE_FEASIBILITY** | 가능성 확인일 뿐이다. |
| 과학 | **SCIENCE_NOT_EVALUATED** | edit loss의 우수성과 신규성은 평가하지 않았다. 입력은 **합성 RGB**이며 자연 이미지 전이가 아니다. |

## A1. 입력 출처 추적 (지시 1)

| 텐서 / 경로 | 출처 | metadata가 모델 입력으로 가는가 |
|---|---|---|
| v0 `render.ProxyEncoder.image_tokens` → `EncGroup.img` → `FactorizedHead` / `HeadScorer` / `ImageOnlyScorer` | 객체마다 shape·color·material·slot embedding의 합(객체 정답과 속성 할당이 token에 들어 있음) | **예 (ORACLE 경로).** `render.PROVENANCE`로 표시했고 지금은 oracle 대조로만 쓴다. |
| v0 `ProxyEncoder.text_tokens` | caption 단어열(realizer 출력). parser를 거치지 않음 | 아니오. `encode_group`의 `parse()`는 assert 검사에만 쓴다. |
| `train.batch_forward` / `torch_head.negative_mask`의 `satisfies(scene, desc)` | oracle truth | 입력은 아니다. **supervision**(false-negative mask)이다. |
| `OracleScorer` | scenes / descs | 설계상 oracle 대조 |
| scene ID(gid), edit(op), render seed | 보고와 부분집합 선택, 렌더링 nuisance | 모델 입력 아님 |
| **v1 이미지**: `pixel_render.render(scene, seed)` → PIL RGB → `ClipEncoder.encode_images` | renderer만 metadata를 읽고, encoder는 PIL 이미지만 받는다(type check) | 아니오 |
| **v1 텍스트**: 캡션 문자열 → `ClipEncoder.encode_texts` | 원문 캡션 | 아니오(parser 정답 없음) |
| `FeatGroup.meta` | Group | label·mask·감사 전용. model scorer가 읽으면 실패하는 poison 테스트로 확인 |

## A2. 실제 명령과 결과 위치

`scripts/ledger_run.py`는 자식 프로세스에 RLIMIT_CPU(남은 예산), RLIMIT_AS 3 GiB, 단일 thread 환경변수를 걸고, 실측 CPU 시간을 ledger에 기록한다.

| 명령 | 결과 위치 | CPU 실측 |
|---|---|---|
| `python3 -m venv .venv`; `.venv/bin/python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu torchvision==0.29.0+cpu`; `.venv/bin/python -m pip install --no-cache-dir numpy pillow open_clip_torch==3.3.0` | `runs/install_pixel_v1_20260926T155648Z` (고정 목록: `requirements-pixel.lock`). 첫 시도 `…155635Z`는 `/usr/bin/time` 부재로 **FAILED**(아무것도 설치되지 않음)이며 보존함. | 설치 43.8 s(wall 49 s). 상한 밖에 별도 기록 |
| `python3 scripts/fetch_encoder.py` | `runs/fetch_encoder_20260926T155833Z` | 605,143,316 B, wall 6.3 s. 별도 기록 |
| `python3 scripts/ledger_run.py --name final_tests_venv_after_manifest_fix -- .venv/bin/python scripts/run_tests.py` | `runs/tests_20260926T162256Z` | 34.9 s |
| `python3 scripts/ledger_run.py --name metadata_check_data_v1 -- python3 scripts/check_metadata.py --config configs/data_v1.json --blind-eval-splits dev,calib` | `runs/metadata_check_data_v1_20260926T161932Z` | 15.4 s |
| `.venv/bin/python scripts/pixel_baseline.py` (**기준 실행**, 커밋 121fdab) | `runs/pixel_baseline_v1_20260926T161957Z` (`metrics.json`, `decisions.json`, `log.txt`, `manifest.json`, `dev_examples.png`, `head_hardneg_seed*.pt`) | 104.5 s, peak RSS 1.63 GiB |
| 같은 명령, 커밋 전의 작업 트리 | `runs/pixel_baseline_v1_20260926T161601Z` | 103.0 s. 결과는 기준 실행과 **동일**(metric, 판정, 파라미터 hash). `POSTHOC_PROVENANCE.json` 참조 |
| debug(작은 panel 4/4/2, 3 step) | `runs/debug_pixel_baseline_tiny_20260926T161518Z` | 22.3 s. 결과는 무의미 |

- **CPU ledger** (`runs/pixel_v1_cpu_ledger.json`): 542.6 / 3,600 s 사용.
  - 여기에는 ledger 이전 개발 작업의 **추정치** 200 s가 포함된다(renderer 감사, 모델 1회 로드, v1 생성 시도; 측정값 아님).
  - 나머지 항목은 probe 9.2 s, 개발 테스트 20.2 s, 테스트 33.1 s이다.
- **worker와 thread**: worker 1개, torch thread 1개. RLIMIT_AS 3 GiB에서 OOM은 없었다.

## A3. 실행한 cell과 실행하지 않은 cell

- **실행**
  - data_v1 생성과 감사.
  - panel 렌더링과 decode 감사(288장).
  - CLIP feature 추출.
  - 대조: oracle, zero-shot CLIP, text-only, image-only(CLIP pooled), random(해석적 값과 MC 500회), shuffle(이미지·캡션), content-only와 binding-only 채널, oracle-object-token 대조.
  - fit sanity(16 그룹, 200 step).
  - hardneg 기준선: 단일 설정, seed 0/1/2, 300 step.
  - train batch에서 gradient 비율 측정.
  - 진단: nuisance ratio, content 잔차, base-vs-edited 검출.
- **NOT_RUN**
  - edit loss(`hardneg_eq`) 학습과 12-setting sweep, `inbatch` 학습.
  - 자연 이미지 전이.
  - data_v1 test 분할(렌더링·인코딩·평가 모두 하지 않음).
  - calibration 임계값 기반 pair accuracy(이번 panel에 calib 없음).
  - encoder와 architecture sweep.
  - v0 12시간 synthetic-feature pilot(승인되지 않았고 철회됨).

## A4. 결과 (dev panel, 기술 관찰이며 가설 검정이 아님)

**조건.**
- dev panel은 32 그룹이고, 그중 binding-necessary(BN, op 종류가 모두 binding 또는 relation)가 16이다.
- seed 3개는 같은 데이터를 공유하므로 독립 표본이 아니다.
- 유의하지 않다는 것을 동등성으로 해석하지 않는다.
- 우연 수준: group 1/6, match 1/2, text·image 1/4. MC 500회로 확인한 값은 0.168 / 0.500 / 0.249 / 0.251이다.

| scorer (입력 출처) | BN group | BN match | BN pair AUC | ALL group | ALL match |
|---|---|---|---|---|---|
| oracle (metadata, 대조) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| zero-shot CLIP (픽셀+캡션, 학습 없음) | 0.000 | 0.688 | 0.497 | 0.312 | 0.844 |
| text-only / image-only (단일 modality) | 0.000 / 0.000 | 0.000 / 0.000 | 0.500 / 0.500 | 0.000 | 0.000 |
| **pixel head (hardneg)**, seed 0/1/2 | 0.062 / 0.000 / 0.000 | 0.625 / 0.500 / 0.438 | 0.508 / 0.500 / 0.508 | 0.375 / 0.156 / 0.188 | 0.812 / 0.750 / 0.719 |
| 같은 head의 content 채널만 | 0 / 0 / 0 | 0.562 / 0.562 / 0.438 | 0.506 / 0.504 / 0.494 | 0.219 / 0.156 / 0.188 | 0.781 / 0.781 / 0.719 |
| 같은 head의 binding 채널만 | 0.125 / 0.062 / 0.000 | 0.625 / 0.625 / 0.562 | 0.515 / 0.537 / 0.511 | 0.062 / 0.062 / 0.000 | 0.500 / 0.562 / 0.562 |
| 같은 head, 이미지를 그룹 간 shuffle | 0 / 0 / 0 | 0.312 / 0.312 / 0.375 | 0.49–0.50 | 0.000 / 0.062 / 0.031 | 0.31–0.44 |
| 같은 head, 캡션을 그룹 간 shuffle | 0 / 0 / 0 | 0.375 / 0.500 / 0.750 | 0.49–0.50 | 0.062 / 0.000 / 0.000 | 0.47–0.66 |
| 같은 head의 train-panel fit | 0.750 / 0.562 / 0.875 | 1.000 (×3) | 0.62–0.72 | 0.844 / 0.781 / 0.938 | 1.000 |
| oracle-object-token 대조 (metadata token, 같은 head와 설정) | 0.312 / 0.125 / 0.188 | 0.812 / 0.750 / 0.750 | 0.619 / 0.548 / 0.569 | 0.312 / 0.094 / 0.188 | 0.906 / 0.781 / 0.875 |

**진단.**
- **nuisance ratio**: 편집에 의한 CLIP pooled 코사인 거리를 같은 장면의 다른 렌더와의 거리로 나눈 값(중앙값)이다. object 4.04, attribute 2.27, binding 1.49, relation 1.04. relation swap은 렌더 nuisance와 거의 같은 크기로만 pooled 표현을 바꾼다.
- **content 잔차**: 같은 seed로 렌더한 BN 쌍에서 ‖Δc‖/‖c‖의 중앙값은 0.23–0.28이다. v0에서는 구조적으로 0이었으므로, **v0의 content 구조적 불변성은 픽셀·CLIP 입력에서 성립하지 않는다.** 그 결과 BN 그룹에서 content 채널의 tie도 사라졌다(tie 비율 0).
- **gradient 비율** ‖∇L_edit‖/‖∇L_task‖ (λ=1, train 첫 batch): init 11.7 / 11.9 / 14.0, 학습 후 0.55 / 0.54 / 0.27. 크기만 보고하며 효용으로 해석하지 않는다.
- **base-vs-edited 검출** (pooled 이미지, train → dev): AUC 0.471, 95% CI [0.31, 0.61], n = 64. 약한 근거이다. data_v1 전체 감사(dev+calib, 1,200 항목)에서는 text 0.508, proxy 이미지 0.512로 PASS였다.
- **fit sanity**: group 1.000인데 aug_group(의미 보존 paraphrase)은 0.125이다. 캡션 표면형을 외운 적합으로 보인다.

**판독** (기술 관찰일 뿐 결론이 아님).
1. 평가 경로가 병목이라는 신호는 없다. oracle은 1.0이고, shuffle 대조는 BN group 0이다.
2. head는 학습 데이터에 적합한다. 그러나 dev BN에서는 group 0.00–0.06, match 0.44–0.63, pair AUC ≈ 0.50으로 우연 수준 부근이거나 그 아래다.
3. dev 전체 점수(ALL group 0.16–0.38)의 대부분은 content 채널만으로도 나온다(0.16–0.22). 즉 binding 없이 풀리는 편집이다.
4. binding이 입력에 이미 들어 있는 oracle-token 대조도 dev BN group이 0.12–0.31에 그친다. 따라서 frozen encoder의 한계(nuisance ratio 1.0–1.5)와, 64 그룹이라는 데이터 크기나 head·설정의 한계를 **분리할 수 없다.**

## A5. 이번 단계의 결정과 수정 (smoke/test 열람 후 수정 포함)

1. **v1 사전등록 범위 수정.** `prereg_pilot_v1`에서 primary endpoint를 binding-necessary 부분집합으로 옮기고 전체 composition 점수는 secondary로 두었다. **smoke_v0를 본 뒤의 범위 수정**이다. v0 test 분할은 smoke에서 열람되었으므로 이제 개발 자료이며, test 크기는 늘리지 않았다. v0 설정과 사전등록은 보존했다.
2. **data_v1.** edit orbit(속성 할당을 뺀 객체·속성 multiset)을 누출 단위로 추가했다. 선착순 registry로는 생성이 불가능해서(orbit 소진) orbit을 hash로 분할에 미리 배정했다. 새 seed를 쓰고 분할 크기는 v0와 같다. data_v0 hash는 불변이다(테스트).
3. **renderer 수정.** 모델을 실행하기 전 decode 감사에서, 금속 원뿔의 반사광이 실루엣 밖에 떨어져 "rubber"로 보이는 의미 손실을 발견했다(1,500 객체 중 16). 반사광 위치를 모양별로 정해 수정한 뒤 6,000 객체에서 불일치 0이다.
4. **debug run의 손실 급등.** 3 step째에 손실이 튀었지만(0.42 → 3.82) 사전 고정 설정은 바꾸지 않았다. 튜닝은 하지 않았다.
5. **첫 stage 실행이 커밋 전 작업 트리에서 수행되었다.** 커밋 후 재실행했고 결과가 완전히 같았다. 두 실행을 모두 보존했다.
6. **manifest dirty 오탐 버그.** 기준 실행 manifest의 `dirty: true`는 오탐이다(목록은 run 산출물인 ledger 1개뿐). 이 버그는 수정하고 회귀 테스트를 추가했다.
7. **입력 bridge.** image·text token을 train panel 통계로 차원별 표준화한다. 실행 전에 고정했고 대안은 시험하지 않았다.

## A6. 미검증 주장과 한계

- **blind scorer의 한계.** 2×2 그룹에서 blind scorer는 group 지표가 구조적으로 0이다. 게다가 각 캡션과 이미지가 한 그룹 안에서 양·음 label을 한 번씩 가지므로 **pair AUC도 구조적으로 정확히 0.5**다. 따라서 blind 대조로는 shortcut의 부재를 보일 수 없다. 정보가 있는 검사는 content-only 채널, shuffle, base-vs-edited 검출(n이 작음), v1 데이터 감사이다.
- **renderer.** 양식화된 2.5D 합성이며, decode 감사는 같은 renderer 계열에 대한 자기 일관성 검사다. CLIP이 "metal/rubber" 같은 단어를 이 렌더링에 대응시키는지는 검증하지 않았다.
- **CLIP 마지막 층 patch token.** 국소화가 약하다고 알려져 있다. 다른 층이나 token 선택은 sweep 금지로 시험하지 않았다.
- **표본 크기.** BN dev는 16 그룹이다. CI가 매우 넓다(예: seed 0의 BN group 95% CI [0, 0.19]).
- **encoder 사용.** encoder는 frozen이다. 어떤 결과도 encoder 자체의 학습 효과로 해석하지 않는다.

## A7. 공개 사항 (supervision, 파라미터, 비용)

- **supervision**: oracle truth label, false-negative mask, 그룹 짝(hard negative). op 종류, base 표시, held-out 정보, scene ID, parser 출력은 입력에 넣지 않았다.
- **학습 파라미터**: 24,586(head). frozen encoder는 151,277,313이다.
- **encoder 비용**: 이미지 288장과 캡션 384개에 50.1 CPU s, 로드 4.5 s.
- **encoder 사용권**: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` rev `1a25a446…`, `open_clip_model.safetensors` sha256 `ac4f8c4b…`. 모델 카드는 MIT이고 연구용이다(영어 전용, 감시·얼굴인식은 범위 밖). open_clip_torch 3.3.0은 MIT이다.
- **의존성 라이선스**: torch는 BSD/Apache 계열, numpy는 BSD-3, Pillow는 MIT-CMU, timm·huggingface-hub·safetensors는 Apache-2.0이다(`importlib.metadata` 기준).

## A8. Blocker와 다음 단계 (승인 필요, 미착수)

- PIXEL_INPUT_BLOCKED는 **해당하지 않는다.**
- **edit loss 비교**: `prereg_pilot_v1`의 seed, 튜닝, CPU 상한을 픽셀 경로에 맞춰 다시 고정하고, calibration panel을 정하고, 승인을 받아야 한다. 현재 A4의 판독상 기준선 자체가 BN에서 일반화하지 않으므로, edit loss를 비교하기 전에 train 크기나 encoder 선택을 먼저 결정해야 할 수 있다(결정 사항).
- **자연 이미지 전이**: 데이터·사용권 승인이 필요하다(B2).
- **LICENSE**: 저장소에 아직 없다(B3).

---

# 1단계 기록 (v0; 인수 기준 `f606795`, 원문 보존)

## 0. 판정 (1단계 v0 당시; 아래 1–8절은 원문 보존)

| 구분 | 상태 | 근거 |
|---|---|---|
| 소프트웨어 | **TECHNICAL_TEST_PASS** | unittest 55/55 통과(skip 0). metadata 감사 FAIL 0. blind 검출 text/image 모두 PASS. smoke 3개 변형의 붕괴 flag 0. |
| 과학 | **SCIENCE_NOT_EVALUATED** | 파일럿은 NOT_RUN. smoke 수치는 변형 간 비교가 금지된 기술 점검용이다. |
| 다음 단계 | ~~READY_FOR_PILOT (조건부)~~ → **철회 (2026-09-26 픽셀 단계)** | v0 image token은 scene metadata에서 직접 만든 oracle 입력이므로 synthetic-feature 파일럿(`prereg_pilot_v0.json`)은 승인되지 않았고 실행하지 않는다. A절 참조. |
| 자연 이미지 전이 | **BLOCKED** | B2 참조. |

## 1. 이번 단계에서 한 일과 멈춘 지점

이 저장소에는 처음에 커밋·파일·데이터·실행 기록이 전혀 없었습니다. 기존 결정이나 ARCHIVE_METHOD도 없었습니다.

| 지시 | 구현 | 검증 |
|---|---|---|
| 1. metadata 수준 정답 생성기 (multiset은 같고 할당만 다른 쌍) | `scene.py`, `ops.py`(`swap_attr`), `groups.py` | `test_binding_swap_preserves_multisets_changes_binding`, `test_degenerate_same_shape_swap_violates_contract`, 모든 그룹의 계약 재검사(`groups_valid`) |
| 2. 객체·속성·관계 변경에 따른 정답 변화 검사 | `oracle.py`(존재 양화, 단사 할당), 연산별 계약(`expected_profile`), 2×2 oracle 표 | `test_captions_oracle.py`(부분 캡션, 관계 불변 쌍, 역관계 등) |
| 3. content/binding head, task/equivariance loss, 붕괴 검출 | `head.py`, `losses.py`, `collapse.py` | finite-difference gradcheck ×3 변형(mutation test로 버그 3종 검출 확인), zero/constant/binding-insensitive 검출 테스트 |
| 4. single-op train과 held-out composition test 분리, 누출 차단 | `dataset.py`: 7개 분할, scene과 caption-content 단위 registry | 감사, 그리고 누출을 주입하는 음성 테스트 4종(scene, paraphrase, held-out pair, template) |
| 5. 같은 데이터·metadata supervision의 hard-negative 기준선 | `train.py`의 `hardneg`(추가로 `inbatch`) | 같은 head와 배치에서 손실만 다름(`docs/RESEARCH_QUESTION.md` 3절) |
| 6. image-only/text-only shortcut, pair/group 평가 | `shortcuts.py`, `evaluate.py` | random은 우연 수준, oracle은 1.0, blind는 group 지표에서 구조적으로 0 |

**여기서 멈췄습니다.** 다운로드, 설치, 유료 API, GPU 학습, 파일럿은 모두 하지 않았습니다.

## 2. 정확한 실행 명령과 실제 결과

모든 실행은 CPU 4코어, Python 3.11.15, 표준 라이브러리만 사용했습니다. 실행 시점의 작업 트리는 clean이었습니다(manifest의 `git.dirty=false`).

| 명령 | 출력 디렉터리 | 커밋 | wall / CPU | peak RSS | 결과 |
|---|---|---|---|---|---|
| `python3 scripts/run_tests.py` | `runs/tests_20260926T152337Z` | a683daa | 11.2 s / 11.2 s | 49 MiB | 55 run, 실패 0, 에러 0, skip 0 |
| `python3 scripts/check_metadata.py --save-data` | `runs/metadata_check_data_v0_20260926T152348Z` | a683daa | 15.1 s / 15.1 s | 488 MiB | 감사 FAIL 0. blind AUC: text 0.5034 [0.4825, 0.5307], image 0.5020 [0.4765, 0.5305], 모두 PASS(허용치 ±0.05) |
| `python3 scripts/smoke_train.py` | `runs/smoke_v0_20260926T152407Z` | a683daa | 64.0 s / 63.9 s | 447 MiB | 3개 변형 완료, 붕괴 flag 없음 |
| `python3 scripts/run_tests.py` | `runs/tests_20260926T152641Z` | b095b16 | 11.6 s / 11.6 s | 49 MiB | 55 run, 실패 0, skip 0 |
| `python3 scripts/smoke_train.py` | `runs/smoke_v0_20260926T152653Z` | b095b16 | 64.9 s / 64.9 s | 448 MiB | 3개 변형 완료, 붕괴 flag 없음. 파라미터 hash가 이전 smoke와 동일(결정적) |

- **hash.**
  - data sha256은 `af7de8f60c227fe2b266ddc1ac226300b70366c4e26cb8e14593c95dd4a93068`이며, 두 실행에서 같았습니다.
  - config sha256은 data `c52ea873…`, encoder `b7228e20…`, smoke `3018cc68…`입니다.
  - smoke head의 param sha256은 inbatch `ed8d0f4c…`, hardneg `6156bcb4…`, hardneg_eq `64ac8e78…`입니다. 전체 값은 각 `manifest.json`에 있습니다.
- **데이터 규모.** train 2000, dev 300, calib 300, 각 test 분할 400 그룹입니다. 모두 3-object 장면이고, 분할마다 op 종류가 균등합니다.
- **생성 데이터 파일.** `data.jsonl`(5.4 MB)은 config와 코드로 결정적으로 재생성되므로 커밋하지 않고 hash만 남겼습니다.
- **smoke 기술 점검값.** 지표 구현 확인용이며 과학적 결과가 아닙니다.
  - oracle: 모든 분할에서 group 1.0, pair AUC 1.0.
  - random: test_composition에서 text 0.22, group 0.14, match 0.45로 우연 수준 부근입니다(n=100).
  - text-only, image-only: text/image/group/match/aug_group 모두 0.
  - content 채널: same-word binding/relation 그룹에서 모든 지표가 0.
  - 이미지 content 잔차(render nuisance 고정): ≤1.05e-15.
- **smoke 수치 해석 금지.** smoke는 seed 1개, 60 step, 평가 부분표본 100 그룹입니다. 변형 간 수치를 비교하거나 결과로 인용하면 안 됩니다.

## 3. 결정 기록

1. **D1. 3-object 장면 고정.** 2-object 캡션 내용 공간은 held-out을 빼면 약 992개입니다. 이 크기로는 누출 없는 분할이 불가능하고, 2-object 비율이 분할마다 달라지는 교란(train 7% 대 test_composition 32%)이 실제로 생겼습니다.
2. **D2. 누출 단위.** scene의 `canonical_key`와 caption 내용(`Desc.canonical`)을 registry에 등록합니다. render variant는 scene을, paraphrase는 caption 내용을 따라 같은 분할에 남습니다. 위치 무관 object-set 중복은 강제하지 않고 정보로만 보고합니다(train과의 중복률: test_iid 0.37, test_heldout_pairs 0.09).
3. **D3. binding 그룹의 단어 규칙.** 객체 목록 구간의 단어 다중집합만 동일하도록 강제합니다. 전체 캡션 동일 여부는 `same_word_multiset` flag로 남기고 부분집합 지표로 보고합니다. 전체를 강제하면 관계절이 항상 편집된 쌍을 가리키는 artifact가 생겼습니다.
4. **D4. 멤버 순서 무작위화.** 그룹의 멤버 0이 항상 원본이 되지 않도록 순서를 섞습니다. base/edited 표시는 감사에만 쓰고 어떤 모델에도 주지 않습니다.
5. **D5. head와 손실.**
   - content는 mean pooling으로 구조적으로 불변이고, binding은 Hadamard 곱입니다.
   - equivariance 손실은 cosine-InfoNCE 형태의 그룹 간 편집 벡터 정렬입니다. 영 해가 최소해가 아니라는 점을 테스트로 확인했습니다.
   - 세 변형은 손실만 다릅니다.
6. **D6. tie 허용치(상대 1e-9).** 부동소수점 반올림 차이가 strict `>` 비교에서 승리로 계산되던 문제를 막습니다.
7. **D7. binding_necessary 부분집합 추가와 사전등록 수정.** 파일럿 실행 전의 수정이며 주 지표는 바꾸지 않았습니다. 4절의 A3을 참조하십시오.
8. **D8. 선행연구와 겹치는 부분은 신규 기여로 주장하지 않습니다.** 상세는 `docs/PRIOR_ART.md` 3절에 있습니다.

## 4. 발견한 artifact와 조치

| # | 발견 | 조치 |
|---|---|---|
| A1 | 분할별 n_objects 분포 교란 | D1로 수정 |
| A2 | 관계절 대상 쌍과 편집 쌍의 상관 | D3으로 수정 |
| A3 | content 채널만으로 test_composition 전체의 GroupMatch가 0.65–0.76(우연 0.5). attribute/object 편집이 섞인 조합은 binding 없이도 일부 풀린다. | `binding_necessary` 부분집합 지표를 추가하고, 사전등록에 "binding" 축소 규칙을 넣었다. |
| A4 | λ=0.3에서 edit-consistency gradient가 task gradient의 약 80배(step 1: |g| 2.41 대 0.031)이고, 60 step 동안 task 손실이 거의 줄지 않음 | 파일럿 λ grid를 0.01–0.3으로 설정. 튜닝 결과로 판정한다. |
| A5 | 부동소수점 tie가 승리로 계산됨 | D6으로 수정 |
| A6 | 이미지별 hard-positive 정확도(`aug`)는 blind-proof가 아님. text-only가 우연(1/3) 근처 또는 그 이상을 얻을 수 있다. | 두 이미지를 모두 요구하는 `aug_group`을 추가(blind는 0) |

## 5. 미검증 주장과 한계

- **proxy feature.** 객체당 깨끗한 token 하나(oracle object token)를 가정합니다. 이미지 쪽 binding은 token 안에서 이미 주어지므로, 실험이 실제로 묻는 것은 텍스트 쪽 binding 학습과 교차 모달 정렬입니다. 픽셀도, 사전학습 encoder도 아닙니다.
- **이미지 binding 채널의 가법성.** 이 채널은 객체 단위로 가법적입니다. 서로 다른 객체를 건드린 조합에서는 구조만으로 합성이 일어납니다(테스트로 확인). held-out 조합 결과를 해석할 때 반드시 분리해야 합니다.
- **blind 검출기의 한계.** 선형 detector(n-gram 로지스틱 회귀, pooled moment)만 썼습니다. PASS라고 해서 비선형 흔적이 없다는 증명은 아닙니다.
- **편집 수준 false negative.** 배치 안에서 서로 다른 그룹이 같은 의미의 편집을 가질 경우를 masking하지 않았습니다.
- **binding_necessary 부분집합의 크기.** test_composition의 400 그룹 중 120입니다. 파일럿에서 CI가 넓어 INCONCLUSIVE가 될 가능성이 큽니다. 파일럿 전에 test 크기를 늘릴지는 연구 책임자가 결정할 사항입니다.
- **held-out 템플릿(T2).** T0와 T1의 국소 구조를 재조합한 템플릿입니다(새 단어 없음). 완전히 새로운 문장 구조의 일반화 시험은 아닙니다.
- **수치 검사.** gradcheck와 mutation test는 구현 검증이며 수학적 증명이 아닙니다.
- **선행연구.** `docs/PRIOR_ART.md` 4절의 미확인 항목이 남아 있습니다. 검색이 완전하지 않으므로 신규성은 인증하지 않습니다.

## 6. Blocker

| # | blocker | 영향 | 해제 조건 |
|---|---|---|---|
| B1 | 환경에 numpy, torch, pytest가 없고, 지시에 따라 설치하지 않았다(`manifest.env.optional_packages_present`에 기록). | 모든 코드가 순수 Python이라 느리다. smoke는 약 0.1 s/step이고, 2000 step 파일럿 1회는 약 3.5분으로 **추정**된다(미측정). | 설치 승인. 합성 파일럿은 현 상태로도 가능하다. |
| B2 | 이미지와 사전학습 encoder 가중치가 없다(자연 이미지, CLIP/OpenCLIP, CounterCurate/VisMin 데이터). | 자연 이미지 전이를 평가할 수 없다. | 다운로드 승인, 그리고 가중치·데이터 사용권 확인(현재 UNVERIFIED). |
| B3 | 저장소에 LICENSE가 없다. | 코드 공개와 재사용 조건이 미정이다. | 소유자 결정 |

## 7. 사용권 확인

| 대상 | 현재 사용 | 상태 |
|---|---|---|
| 코드 | 이 저장소 코드만 사용. 외부 코드 없음 | LICENSE 없음(B3) |
| 데이터 | `bindcomp`가 생성한 합성 metadata만 사용 | 제3자 데이터 없음 |
| 가중치 | 사전학습 가중치 없음. head는 여기서 난수로 초기화해 학습 | 제3자 가중치 없음 |
| encoder | 합성 proxy(`render.py`, 고정 난수 embedding) | 제3자 encoder 없음 |
| 향후 후보: CLIP/OpenCLIP 가중치, CounterCurate/VisMin 데이터 | 미사용 | **UNVERIFIED**. 사용 전에 확인해야 함 |

## 8. 다음 단계 (승인 필요, 현재 미착수)

1. `configs/prereg_pilot_v0.json` 승인 또는 수정: 최소관심효과, seed, 튜닝 grid, 자원 상한, 그리고 binding_necessary 검정력을 위한 test 크기 확대 여부.
2. 승인되면 사전등록대로 합성 파일럿을 실행합니다. CPU로 약 5–6시간(추정)이며, 실패·OOM·NOT_RUN은 모두 보존합니다.
3. 자연 이미지 전이는 별도 승인이 필요합니다(B2). encoder를 freeze할 경우 encoder 자체의 학습 효과로 설명하지 않습니다.
