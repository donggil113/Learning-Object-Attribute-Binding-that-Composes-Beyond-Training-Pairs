# STATUS: P3 Learning Object-Attribute Binding that Composes Beyond Training Pairs

마지막 갱신: 2026-09-26. 기준 커밋은 `b095b16`입니다(`c786999`, `a683daa` 이후).

## 0. 판정 (소프트웨어와 과학을 분리)

| 구분 | 상태 | 근거 |
|---|---|---|
| 소프트웨어 | **TECHNICAL_TEST_PASS** | unittest 55/55 통과(skip 0). metadata 감사 FAIL 0. blind 검출 text/image 모두 PASS. smoke 3개 변형의 붕괴 flag 0. |
| 과학 | **SCIENCE_NOT_EVALUATED** | 파일럿은 NOT_RUN. smoke 수치는 변형 간 비교가 금지된 기술 점검용이다. |
| 다음 단계 | **READY_FOR_PILOT (조건부)** | 적용 범위는 합성 proxy feature 파일럿에 한한다. `configs/prereg_pilot_v0.json`을 연구 책임자가 승인해야 한다. 신규성 인증이나 채택 가능성 확인이 아니다. |
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
