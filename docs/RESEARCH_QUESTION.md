# 연구 질문, 반증 조건, 정보 접근, 목적함수 가정 (고정: 2026-09-26)

> **2026-09-26 갱신.**
> - 아래 v0 질문이 쓰는 image feature는 scene metadata에서 만든 oracle token이다. 따라서 v0 synthetic-feature 파일럿은 철회했다.
> - 픽셀 입력 경로와 기준선 실행 결과는 `STATUS.md` A절에 있다.
> - 향후 질문은 `configs/prereg_pilot_v1.json`(NOT_RUN)이다. binding-necessary 부분집합이 primary이며, smoke 이후의 범위 수정임을 명시한다.


## 1. 연구 질문 (단일, 반증 가능)

**RQ.** 모든 변형은 같은 frozen token feature, 같은 content/binding head, 같은 단일 편집 학습 그룹, 같은 false-negative masking을 씁니다. 이 조건에서 hard-negative contrastive 목적함수에 **그룹 간 edit-consistency 손실**을 더하면 결과가 달라지는지 묻습니다. edit-consistency 손실은 이미지 편집 벡터와 텍스트 편집 벡터를 cosine-InfoNCE로 정렬합니다. 구체적으로, 학습에 없던 **두 편집의 조합**(`test_composition`)에서 Winoground group score가 같은 데이터로 학습한 hard-negative 기준선(`hardneg`)보다 **절대값 0.05 이상** 높아지는가?

- 처치: `hardneg_eq`. 비교군: `hardneg`(같은 데이터, 같은 metadata supervision). 보조 비교군: `inbatch`.
- 주 지표, 최소관심효과, seed, 튜닝 범위, 자원 상한은 `configs/prereg_pilot_v0.json`에 고정했습니다. 상태는 **NOT_RUN**입니다.

## 2. 판정 규칙

| 판정 | 조건 (주 지표 차이의 95% paired bootstrap CI, test seed 5개 평균) |
|---|---|
| SUPPORTED | 효과 ≥ 0.05이고 CI 하한 > 0 |
| REJECTED | CI 상한 < 0.05 |
| INCONCLUSIVE | 그 외. 이 경우 "효과 없음"이라고 쓰지 않는다. |

**주장을 축소하는 규칙** (SUPPORTED가 나와도 적용합니다):

| 규칙 | 조건 | 축소 내용 |
|---|---|---|
| 문장 규칙 의존 | `test_heldout_template` 효과 < 주 효과의 절반 | 본 템플릿에 한정된 주장 |
| metric 의존 | group과 GroupMatch 중 한쪽에서만 SUPPORTED | metric 의존 주장 |
| 축 한정 | `test_composition`과 `test_heldout_pairs` 중 한쪽에서만 효과 | 해당 축에 한정 |
| 구조 기인 | 효과가 서로 다른 객체를 건드린 조합에만 있음 | 이미지 head의 구조적 가법성(아래 4절)과 텍스트 쪽 학습의 결과로 기술 |
| binding 한정 | `test_composition` 중 binding_necessary 부분집합(op 종류가 모두 binding 또는 relation)에서 효과 < 0.05이거나 CI 하한 ≤ 0 | "binding"이 아니라 일반 편집 조합의 결과로 기술. smoke_v0에서 content-only 채널이 전체 `test_composition`의 GroupMatch 0.65–0.76(우연 0.5)에 도달했기 때문에 추가한 규칙이다. 파일럿 실행 전의 수정이며 처치-기준선 차이는 보지 않았다(`amendments` 참조). |
| 데이터 흔적 | blind detectability가 WARN이거나, content-only 채널이 same-word binding 그룹에서 group > 우연 + 0.05 | 데이터부터 수정하고, 그 전에는 주장하지 않음 |
| 범위 | 항상 | 합성 proxy feature 결과일 뿐이다. 자연 이미지 전이나 encoder 자체의 학습 효과는 주장하지 않는다(encoder는 frozen이고 proxy다). |

## 3. 비교군의 정보 접근

| 변형 | 입력 쌍 | 그룹 짝을 negative로 사용 | 그룹 짝으로 편집 벡터 구성 | oracle truth (false-negative mask) | op 종류 / base-edited 표시 / held-out 정보 | 학습 파라미터 | step당 계산 |
|---|---|---|---|---|---|---|---|
| `inbatch` | 두 멤버 모두 | 아니오(mask) | 아니오 | 예 | 아니오 | 1162 | 1× (측정값은 manifest) |
| `hardneg` | 두 멤버 모두 | 예 | 아니오 | 예 | 아니오 | 1162 | ≈1× |
| `hardneg_eq` | 두 멤버 모두 | 예 | 예 | 예 | 아니오 | 1162 | edit 손실이 추가됨. smoke 60 step 기준 inbatch 5.96 s / hardneg 6.39 s / hardneg_eq 6.33 s로, 1회 측정이라 차이를 주장할 수 없음 |
| `text_only` | 캡션 + oracle label | — | — | label | 아니오 | n-gram 가중치 | — |
| `image_only` | proxy 이미지 feature + oracle label | — | — | label | 아니오 | pooled moment 가중치 | — |
| head `[content]` / `[binding]` | 학습된 head에서 한 채널만 남긴 평가 시점 ablation | — | — | — | — | — | — |

- 세 학습 변형의 차이는 **손실뿐**입니다. head, 초기화 seed, 배치 순서, optimizer, step 수, 평가 코드는 같습니다.
- `hardneg_eq`가 추가로 쓰는 것은 "같은 그룹의 두 멤버"라는 짝 정보뿐입니다. `hardneg`도 이 짝을 hard negative로 이미 씁니다. 추가 label이나 op 종류는 쓰지 않습니다.
- step cap이 같다고 실지출이 같지는 않습니다. 실제 wall/CPU 시간은 run manifest의 `timing`에 기록합니다.

## 4. 목적함수와 구조의 가정 (알려진 한계 포함)

1. **InfoNCE.** 행과 열마다 정답이 하나라고 가정합니다. oracle이 참으로 판정한 대각 밖 쌍은 negative에서 제외할 뿐, positive로 쓰지 않습니다.
2. **edit-consistency.**
   - 같은 편집이 이미지와 텍스트에서 만드는 변화가 선형(cosine) 정렬 가능하다고 가정합니다.
   - 배치 안의 다른 그룹 편집은 모두 서로 다르다고 가정합니다(편집 수준 false negative를 masking하지 않음).
   - 영(0) 벡터는 이 손실의 최소해가 아닙니다(손실 = log B). 반면 naive `delta_mse`는 영 해에서 0이 됩니다. 둘 다 테스트로 확인했습니다.
3. **그룹 내 항등식.** s00 + s11 − s01 − s10 = ⟨ΔI, ΔT⟩입니다. 따라서 그룹 **내부**의 Δ 정렬은 hard-negative 손실과 GroupMatch가 이미 다룹니다. 새로운 신호는 **그룹 간** 대조뿐입니다.
4. **content 불변성은 구조로 보장되며 학습되지 않습니다.**
   - content는 mean pooling이므로 binding/relation 편집에 정확히 불변입니다(render nuisance를 고정했을 때 잔차 ~1e-16).
   - binding 채널은 content 변화에도 반응합니다. 즉, "분리"는 content 쪽에서만 보장됩니다.
5. **이미지 binding 채널은 객체 단위로 가법적입니다.** 따라서 서로 다른 객체를 건드린 두 편집의 Δb는 이미지 쪽에서 **정확히** 합성됩니다. held-out 조합 결과의 일부는 학습이 아니라 구조에서 나올 수 있습니다(테스트 `test_image_binding_additive_over_disjoint_edits`).
6. **oracle object token.** proxy 이미지 feature는 객체당 token 하나를 주고, 각 token 안에 그 객체의 속성이 이미 묶여 있습니다. 이미지 쪽 binding 문제는 사실상 풀린 상태로 주어지고, 실험이 묻는 것은 주로 **텍스트 쪽 binding 학습과 교차 모달 정렬**입니다. 실제 encoder는 이런 token을 주지 않습니다.
7. **frozen proxy encoder.** 학습되는 것은 head뿐입니다. 결과를 encoder의 학습 효과로 해석하지 않습니다.
