# 선행연구 대조 및 주장 표 (P3)

검증 일자: 2026-09-26.

**검증 방법.** 모든 항목의 제목·저자·연도·ID는 arXiv 초록 페이지에서 확인했습니다. 본문은 arXiv HTML이나 ar5iv에서 해당 절을 직접 읽은 경우에만 `FULL_TEXT`로 표시했습니다. 초록이나 서론만 읽었다면 `FULL_TEXT_UNVERIFIED`, 일부 절만 읽었다면 `PARTIAL`입니다. 조회의 일부는 AI 하위 에이전트가 수행했습니다(AI_USAGE.md 참조). 주 에이전트는 두 가지를 직접 다시 확인했습니다. 하나는 8개 arXiv ID의 제목·날짜이고, 다른 하나는 TTM의 GroupMatch 정의(TeX 원문)입니다.

## 1. 가장 가까운 원논문 (8편)

| # | 논문 | 저자, 연도, 게재처 | ID | 본문 | 이번 과제와의 관계 (확인한 사실) |
|---|---|---|---|---|---|
| 1 | CounterCurate: Enhancing Physical and Semantic Visio-Linguistic Compositional Reasoning via Counterfactual Examples | Zhang, Cai, Xie, Lee. 2024, Findings of ACL | 2402.13254 | FULL_TEXT | GLIGEN과 DALLE-3로 counterfactual 이미지를 만들고, GPT-4V로 부정 캡션(noun·adjective 교체, 구 swap)을 만든다. CLIP은 (C,I,C′,I′)를 한 배치로 묶는 "Grouping"으로 학습하며, 손실은 원래의 대각 cross-entropy를 그대로 쓴다. |
| 2 | VisMin: Visual Minimal-Change Understanding | Awal, Ahmadi, Zhang, Agrawal. 2024, NeurIPS | 2407.16772 | FULL_TEXT | 변화 유형은 object/attribute/count/spatial relation 네 가지이며 한 번에 하나만 바꾼다. 벤치마크는 인간이 검증하고, 학습 데이터는 자동 필터만 거친다. 편집 쌍을 hard negative로 추가하되 손실은 수정하지 않는다. Winoground식 text/image/group score를 쓴다. |
| 3 | Test-Time Matching: Unlocking Compositional Reasoning in Multimodal Models | Zhu, Zhang, Tang. 2025, ICLR 2026 예정(arXiv 주석) | 2510.07632 | FULL_TEXT (GroupMatch 정의는 TeX 원문을 직접 확인) | GroupScore가 능력을 과소평가한다고 주장하고 GroupMatch를 제안한다(k=2일 때 s11+s22 > s12+s21, 우연 수준 1/k!). TTM은 **비라벨 test group**의 pseudo-label로 반복 fine-tune하는 transductive 방법이다. |
| 4 | Equivariant Similarity for Vision-Language Foundation Models (EqSim, EqBen) | Wang, Lin, Li, Lin, Yang, Zhang 외. 2023, ICCV | 2303.14465 | FULL_TEXT | 두 matched pair의 **스칼라 유사도**에 동변성 제약을 건다(v1: s12=s21, v2: s11−s12=s22−s21). counterfactual 데이터 없이 임의의 두 학습 쌍에 적용된다. |
| 5 | Winoground | Thrush 외. 2022, CVPR | 2204.03162 | FULL_TEXT | text/image/group score 정의. 우연 수준은 25/25/16.67이다. |
| 6 | When and why vision-language models behave like bags-of-words (ARO, NegCLIP) | Yuksekgonul 외. 2023, ICLR | 2210.01936 | FULL_TEXT | retrieval 목적함수는 순서·결합 정보 없이도 낮은 손실에 도달할 수 있다. NegCLIP은 swap으로 부정 캡션을 만든다. |
| 7 | Does CLIP Bind Concepts? Probing Compositionality in Large Image Models | Lewis, Nayak, Yu, Yu, Merullo, Bach, Pavlick. 2024, Findings of EACL | 2212.10537 | FULL_TEXT | CLEVR식 합성 데이터에서 속성-객체 쌍 일부를 generalization split로 hold-out한다. frozen CLIP 위에 Add/Mult/Conv/TL/RF head를 학습한다. two-object gen에서 CLIP-FT는 0.25, RF는 20.36이다. |
| 8 | Object-centric Binding in Contrastive Language-Image Pretraining (OC-CLIP) | Assouel, Astolfi, Bordes, Drozdzal, Romero-Soriano. 2025, NeurIPS 2025 poster | 2502.14113 | FULL_TEXT (unseen-pair 수치는 그림에만 있어 UNVERIFIED) | 텍스트 scene graph 노드가 query인 binding module과 L_itc + L_rel을 쓰며, parser가 주어진다고 가정한다. PUG 합성 데이터와 자연 데이터 결과를 보고한다. |

## 2. 최신 후속 연구와 관련 기반

| 논문 | ID | 본문 | 관계 |
|---|---|---|---|
| CLIP Behaves like a Bag-of-Words Model Cross-modally but not Uni-modally (LABCLIP). Koishigarina, Uselis, Oh. ICLR 2026 | 2502.03566 | FULL_TEXT | frozen CLIP 텍스트 임베딩에 선형 사상을 학습하고, 개념을 permute해서 부정 예를 만든다. "binding은 이미 unimodal 임베딩에 있다"고 주장한다. **이번 head 접근의 가장 가까운 선행**이다. |
| How can embedding models bind concepts? Uselis, Koishigarina, Oh. ICML 2026 | 2605.31503 | FULL_TEXT_UNVERIFIED (5절과 그림 캡션만) | 일반화하는 모델은 개념 간 곱셈적(multiplicative) 상호작용을 구현한다고 주장한다. 이번 Hadamard binding head가 이 계열에 속한다. |
| Common Data Properties Limit Object-Attribute Binding in CLIP. Gurung, Hoffmann, Brox. GCPR 2025 | 2507.07985 | PARTIAL | 합성 데이터에서 batch를 키우거나 hard negative를 명시해도 binding이 학습되지 않는다고 보고한다. |
| Does Data Scaling Lead to Visual Compositional Generalization? Uselis, Dittadi, Oh. ICML 2025 | 2507.07102 | FULL_TEXT_UNVERIFIED | 규모가 아니라 데이터 다양성이 구성적 일반화를 결정한다고 주장한다. |
| SugarCrepe. Hsieh 외. NeurIPS 2023 D&B | 2306.14610 | FULL_TEXT | 기존 벤치마크 10개 과제 중 9개에서 blind(텍스트 전용) 모델이 최고 성능을 낸다. 이번 blind 검사의 근거다. |
| Revisiting the Role of Language Priors in VLMs. Lin 외. ICML 2024 | 2306.01879 | FULL_TEXT | P(text) 같은 blind 해법이 ARO 등을 상당 부분 푼다. |
| The Hard Positive Truth about VL Compositionality. Kamath 외. ECCV 2024 | 2409.17958 | FULL_TEXT | hard negative fine-tuning이 hard positive에서 성능을 떨어뜨린다. 이번 `aug` 지표의 근거다. |
| CE-CLIP. Zhang, Awal, Agrawal. CVPR 2024 | 2306.08832 | FULL_TEXT (L_imc 식은 PDF 재확인 필요) | 손실은 L_itc(hn) + L_imc + L_cmr이다. hard-negative 계열 기준선의 참고로 쓴다. |
| CREPE. Ma 외. CVPR 2023 | 2212.07796 | FULL_TEXT | seen/unseen compound(SC/UC/UA) 분할 정의. |
| CLEVR (CoGenT 조건 A/B). Johnson 외. CVPR 2017 | 1612.06890 | FULL_TEXT | 조건 A와 B 사이에서 shape-color 팔레트를 교환한다. held-out pair 설계의 원형이다. |
| CSP (compositional zero-shot). Nayak, Yu, Bach. ICLR 2023 | 2204.03574 | FULL_TEXT | seen과 unseen attribute-object 조합은 서로소다. |
| Slot Attention. Locatello 외. NeurIPS 2020 | 2006.15055 | FULL_TEXT | 입력 순열에 불변이고 slot 순열에 동변이다. |
| Equivariant Contrastive Learning. Dangovski 외. ICLR 2022 | 2111.00899 | FULL_TEXT | 일부 변환에는 동변성을, 나머지에는 불변성을 부여한다. |
| Group Equivariant CNNs. Cohen & Welling. ICML 2016 | 1602.07576 | FULL_TEXT | 동변성 Φ(T_g x) = T′_g Φ(x)를 정의하며, 불변성은 그 특수한 경우다. |
| Deep Sets. Zaheer 외. NeurIPS 2017 | 1703.06114 | FULL_TEXT | 순열 불변 함수는 ρ(Σφ(x)) 꼴로 쓸 수 있다. 이번 content head(mean pooling)가 이 형태다. |
| VICReg. Bardes, Ponce, LeCun. ICLR 2022 | 2105.04906 | FULL_TEXT | collapse를 막는 variance 항. 붕괴 검출기 설계의 참고다. |
| Tensor product variable binding. Smolensky. AI 46, 1990 | DOI 10.1016/0004-3702(90)90007-M | FULL_TEXT_UNVERIFIED | 서지 정보만 확인했다. |
| StyleGAN-NADA. Gal 외. | 2108.00946 | FULL_TEXT (게재처 UNVERIFIED) | 방향 손실 1 − cos(ΔI, ΔT). CLIP은 frozen이고 generator를 학습한다. |
| PC-CLIP: Finetuning CLIP to Reason about Pairwise Differences. Sam 외. 2024 | 2409.09721 | FULL_TEXT (게재처 UNVERIFIED) | 이미지 임베딩 차이를 "차이를 설명한 텍스트"의 임베딩에 맞춘다. |
| DiCE-CIR. Na, Kim, Lee. 2026 | 2607.04665 | 방법 절만 | 교차 모달 residual 정렬. composed retrieval 모듈을 학습한다. |

## 3. 주장 표: 무엇이 이미 알려졌는가

| 구성 요소 | 이미 알려짐 (근거) | 이번 저장소에서의 지위 |
|---|---|---|
| counterfactual / minimal-change 쌍 생성 | CounterCurate, VisMin, NegCLIP, CE-CLIP, (SPARCL, TripletCLIP: 초록 수준) | **신규 기여 아님.** metadata 수준 생성기는 실험 도구다. |
| 그룹 짝을 hard negative로 쓰는 contrastive 학습 | NegCLIP, CounterCurate "Grouping", VisMin-CLIP, CE-CLIP | **같은 데이터로 학습한 기준선(`hardneg`)**으로 사용한다. |
| 합성 장면에서 attribute-object 쌍 hold-out | CLEVR 조건 A/B, Lewis 외 2024, LABCLIP, Gurung 외, OC-CLIP(PUG) | **신규 아님.** 평가 설계로 차용했다. |
| frozen encoder 위의 작은 binding head | Lewis 외(Add/Mult/TL/RF), LABCLIP(선형 사상) | **신규 아님.** 이번 head는 그 변형이다. |
| 곱셈적 상호작용(Hadamard/TPR 계열) | Smolensky 1990, Lewis 외(RF/TL), Uselis 외 2026(UNVERIFIED) | **신규 아님.** |
| 최소쌍의 **유사도 수준** 동변성 제약 | EqSim | **신규 아님.** |
| 이미지·텍스트 **임베딩 차이 벡터**의 방향 정렬 | StyleGAN-NADA(frozen CLIP으로 generator 학습), PC-CLIP(텍스트 쪽은 차이 설명문), DiCE-CIR(CIR) | 손실 **형태**는 알려져 있다. 이를 binding 학습에 쓰고 held-out 편집 조합에서 평가한 선행은 **확인하지 못했다.** 검색이 완전하지 않으므로 **신규성은 인증하지 않는다.** |
| 그룹 내 Δ 정렬 | 대수 항등식 s00 + s11 − s01 − s10 = ⟨ΔI, ΔT⟩. hard-negative 손실과 GroupMatch가 이미 이 양을 다룬다. | **기여 아님.** `edit_consistency`에서 새로운 신호는 **그룹 간** 대조뿐이다(`bindcomp/losses.py` docstring). |
| text/image/group score, GroupMatch, hard-positive 지표, blind 기준선 | Winoground, TTM, Kamath 외, SugarCrepe, Lin 외 | 평가 도구로 사용한다. |
| object-centric slot | Slot Attention, OC-CLIP | 사용하지 않는다. 대신 **oracle object token 가정**을 쓴다(STATUS 한계 참조). |

**결론.** 현 단계에서 어떤 구성 요소도 신규 기여로 주장하지 않습니다. 남은 것은 `docs/RESEARCH_QUESTION.md`의 경험적 질문 하나이며, 그 답이 긍정이어도 기존 방법의 조합일 수 있습니다.

## 4. 확인하지 못한 항목

- TTM이 pseudo-label로 fine-tune할 때 쓰는 손실의 이름(3절과 부록 B.1.1에 명시되지 않음).
- OC-CLIP의 unseen-pair 수치(그림에만 있음), parser 모델 표기의 불일치(부록 A.3과 A.6).
- CE-CLIP L_imc 식의 정확한 형태(HTML 렌더링만 봄).
- VisMin의 NeurIPS 트랙 구분.
- Campbell 외(NeurIPS 2024, 2411.00238)의 실험 세부. 초록과 서론만 읽었다.
- Kang 외(ICCV 2025, 2503.08723). 초록만 읽었다.
- Pham 외(2603.25722), CPI(2605.22651), ABE-CLIP(2512.17178), Auto-Comp(2602.02043), Lo(2608.15971). 초록이나 검색 결과 수준으로만 확인했다.
