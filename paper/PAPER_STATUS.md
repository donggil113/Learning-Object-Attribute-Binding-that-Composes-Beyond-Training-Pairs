# PAPER_STATUS: Working Draft

| 항목 | 상태 |
|---|---|
| 버전 | **v0**: 기존 결과만 사용(small panel 기준선, data v1 감사, v0 smoke 관찰) |
| 제목 (가제) | Evaluating Compositional Binding from Pixels without Oracle Features |
| 형식 | 공식 ICLR 2027 style(ZIP sha256 `0d940dfa…`, 파일 미수정). 익명 모드. header는 `main.tex`에서 "Internal working draft --- not submitted"로 교체(BUILD.md) |
| 빌드 | pdfTeX (TeX Live 2023/Debian) 컴파일 성공. 미정의 참조 0, overfull box 0 |
| 쪽수 | PDF 11쪽. 본문(Conclusion까지)은 7쪽에서 끝남. 한도 9쪽 |
| 숫자 출처 | 모든 수치는 `generated/numbers.tex` 매크로이며, 출처 매핑은 `generated/provenance.tsv` |
| 인용 | arXiv BibTeX export 25건 + OpenCLIP software 1건. venue note는 검증된 것만 |
| 사람 검토 | **HUMAN_REVIEW_PENDING**. 저자·검토자 없음 |
| 제출 상태 | 미제출. 제출 ID·공개 URL 없음 |

## 섹션별 작성 상태
- Abstract, Introduction, Related Work, Setting, Objectives, Experiments(5.1–5.3), Discussion/Limitations, Conclusion, AI use, Reproducibility: 모두 완전한 문장으로 작성.
- 5.4 paired comparison: v0에서는 **NOT RUN**으로 명시(Appendix E의 계획).
- Appendix A–F 작성: chance 유도, split, blind 구조 증명, 전체 표, Planned (NOT RUN), provenance.

## 남은 주장과 증거 상태
`claim_evidence.tsv` 참조.
- 원자료로 검증: C04–C06, C08–C11, C13–C15, C19, C21.
- 유도: C16–C18. 표준 결과이며 새 정리로 주장하지 않는다.
- 해석: C12.
- 절차상 주장: C20.
- 미실행: C24.
