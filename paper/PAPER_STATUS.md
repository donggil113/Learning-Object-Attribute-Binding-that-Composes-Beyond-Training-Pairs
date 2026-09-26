# PAPER_STATUS: Working Draft

| 항목 | 상태 |
|---|---|
| 버전 | **v1**: v0(커밋 `6d6887e`) 이후 split 의미 분석과 paired 비교 결과를 반영 |
| 제목 (가제) | Evaluating Compositional Binding from Pixels without Oracle Features |
| 형식 | 공식 ICLR 2027 style을 수정 없이 사용(ZIP sha256 `0d940dfa…`). 익명 모드. header만 `main.tex`에서 "Internal working draft --- not submitted"로 교체(BUILD.md). **ICLR_FORMAT_VERIFIED_FILES**: 파일은 공식 ZIP과 hash가 같다. 단, 공식 제출 PDF가 아닌 내부 초안이다 |
| 빌드 | pdfTeX (TeX Live 2023/Debian) 컴파일 성공. 미정의 참조 0, overfull box 0 |
| 쪽수 | PDF 13쪽. 본문(Conclusion까지)은 **8쪽**에서 끝남(한도 9쪽). 참고문헌은 9–10쪽, 부록은 10–13쪽 |
| 숫자 출처 | 모든 결과 수치는 `generated/numbers.tex` 매크로(총 327개)이며, 출처는 `generated/provenance.tsv`. 표 4개는 exporter가 생성 |
| 인용 | arXiv BibTeX export 25건 + OpenCLIP software 1건. venue note는 검증된 것만 |
| 사람 검토 | **HUMAN_REVIEW_PENDING**. 저자·검토자 없음 |
| 제출 상태 | 미제출. 제출 ID·공개 URL 없음 |

## 섹션별 작성 상태
- Abstract, Introduction, Related Work, Setting, Objectives(표준 항등식을 Remark로 표기), Experiments 5.1–5.4, Discussion/Limitations, Conclusion, AI use, Reproducibility: 모두 완전한 문장으로 작성.
- 5.4 paired 비교: 사전 고정 규칙에 따른 판정 `OPTIMIZATION_OR_INPUT_UNRESOLVED`를 부정적·불확정 결과 그대로 본문에 서술.
- Appendix
  - A: chance 유도
  - B: split 의미와 metadata 표
  - C: blind 구조 증명
  - D: all-kinds 표
  - E: Planned (NOT RUN). 안정화 재비교, test, 자연 이미지, 다른 encoder·층, calibrated pair accuracy, sweep
  - F: provenance

## 남은 주장과 증거 상태
`claim_evidence.tsv`(35건) 참조.
- 원자료로 검증: C04–C06, C08–C11, C13–C15, C19, C21, C24–C27, C30–C34.
- 유도: C16–C18. 표준 결과이며 새 정리가 아니다.
- 해석(검정 없음): C12, C28.
- 설명 가능성만 제시하고 미검정: C29의 메커니즘.
- 절차상 주장: C20, C35.

## 알려진 문제 (사람 검토 필요)
- arXiv BibTeX export의 `year`는 최신 버전 기준이다(예: TTM 2026, LABCLIP 2026). 인용 연도 정책은 저자가 결정해야 한다.
- `uselis2026bind`, `gurung2025binding`은 본문을 부분적으로만 확인했다(docs/PRIOR_ART.md).
- 본문의 "correlated preferences" 설명(C29)은 가능한 기제일 뿐 검증하지 않았다.
