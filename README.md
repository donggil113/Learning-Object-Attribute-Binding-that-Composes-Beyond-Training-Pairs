# P3 — Learning Object-Attribute Binding that Composes Beyond Training Pairs

현재 단계는 **metadata 수준 정답 생성기, 검사, 최소 head 구현**까지이며 여기서 멈춥니다. 파일럿은 실행하지 않았습니다(`configs/prereg_pilot_v0.json`, NOT_RUN). 현재 상태는 [STATUS.md](STATUS.md)를 보십시오.

- 연구 질문과 반증 조건: [docs/RESEARCH_QUESTION.md](docs/RESEARCH_QUESTION.md)
- 선행연구 대조와 주장 표: [docs/PRIOR_ART.md](docs/PRIOR_ART.md)
- AI 사용 내역: [AI_USAGE.md](AI_USAGE.md)

## 요구 환경

Python 3.11 표준 라이브러리만 씁니다. 이 환경에는 numpy, torch, pytest가 없고, 지시에 따라 설치하지 않았습니다. 테스트는 `unittest`로 돌립니다.

## 실행

```bash
python3 scripts/run_tests.py                  # 전체 테스트와 manifest (runs/tests_*)
python3 scripts/check_metadata.py --save-data # 데이터 생성, 누출·계약 감사, blind 검출 (runs/metadata_check_*)
python3 scripts/smoke_train.py                # 기술 smoke (파일럿 아님) (runs/smoke_v0_*)
```

## 구성

| 경로 | 내용 |
|---|---|
| `bindcomp/scene.py`, `ops.py` | scene graph, 편집 연산 4종(binding swap, attribute·object 교체, relation swap)과 각 연산의 계약 |
| `bindcomp/captions.py`, `oracle.py` | 기술 그래프, 템플릿 3종, realize/parse, 존재 양화 정답 판정 |
| `bindcomp/groups.py` | 2×2 최소쌍 그룹, 계약 검사(`check_invariants`), oracle 표 검사, 멤버 순서 무작위화 |
| `bindcomp/dataset.py` | 분할 7종, scene·caption-content 단위 누출 방지 registry, 감사 |
| `bindcomp/render.py`, `encode.py` | **proxy** feature(합성, frozen). 실제 렌더러나 encoder가 아님 |
| `bindcomp/head.py` | content/binding head (수동 gradient) |
| `bindcomp/losses.py`, `train.py`, `optim.py` | InfoNCE(in-batch / hard-negative), edit-consistency, Adam |
| `bindcomp/collapse.py` | ZERO / CONSTANT / LOW_RANK / BINDING_INSENSITIVE / CONTENT_LEAK 검출 |
| `bindcomp/evaluate.py`, `shortcuts.py` | pair·group 평가(text/image/group/GroupMatch/hard-positive), blind 기준선 |
| `bindcomp/manifest.py` | git, config, data, model hash, 환경, 실행시간, peak RSS |
| `configs/` | data, encoder, smoke 설정과 파일럿 사전등록 |
| `runs/` | raw log와 manifest (생성 데이터는 hash만 보관) |
