# Build notes (working draft)

## Status
- Format: official ICLR 2027 LaTeX style, anonymous (non-final) mode. `\iclrfinalcopy` is NOT used.
- One deliberate deviation from the style's default output, made in `main.tex` and not in any style file:
  the non-final mode prints the header "Under review as a conference paper at ICLR 2027". This draft has
  not been submitted, so `main.tex` calls `\lhead{Internal working draft --- not submitted (ICLR 2027 style)}`
  right after `\maketitle`. A status box under the author block explains that the style's author line
  "Paper under double-blind review" does not indicate a submission.
- No submission ID, acceptance status or public anonymous URL is used.

## Official style files
- Source: https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip, linked from
  https://iclr.cc/Conferences/2027/AuthorGuidelines (fetched 2026-09-26).
- ZIP sha256: `0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`.
- Files copied unmodified into `paper/`. Their sha256 values are identical to the ZIP contents:

| file | sha256 |
|---|---|
| iclr2027_conference.sty | 797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6 |
| iclr2027_conference.bst | 2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5 |
| fancyhdr.sty | b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea |
| natbib.sty | 88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d |
| math_commands.tex | 90473c4d0542070db244cea73ef962d6cddc5b2a746757e6a40ddf5fdfb90ba9 |

- The guideline page states: main text at most 9 pages at submission; references and appendices do not
  count; an AI use statement is required and does not count; a reproducibility statement is recommended.

## Toolchain
- The container had no TeX engine. Installed from Ubuntu 24.04's signed repositories
  (`apt-get install --no-install-recommends`):
  texlive-latex-base, texlive-latex-recommended, texlive-fonts-recommended (TeX Live 2023/Debian,
  pdfTeX 3.141592653-2.6-1.40.25), plus dependencies.
  Log: `runs/install_latex_20260926T223844Z/`.
- poppler-utils 24.02.0 was installed for page rendering checks. The first attempt failed on a stale
  package index and is kept; the retry, after `apt-get update`, succeeded
  (`runs/install_poppler_*`).
- These installs change the ephemeral container's system packages. The Python environment
  (`.venv`, `requirements-pixel.lock`) was not touched. The installs are recorded separately from both
  CPU caps.
- `cleveref` is not in the installed set, so the text uses plain `\ref`.

## Build
```bash
python3 scripts/ledger_run.py --name paper_build --ledger runs/paper_build_cpu_ledger.json --cap-s 600 -- bash paper/build.sh
```
`paper/build.sh` runs three steps:
1. `scripts/export_paper_numbers.py` regenerates `paper/generated/numbers.tex`,
   `paper/generated/provenance.tsv` and `paper/tables/*.tex` from the raw run files.
2. `scripts/make_paper_figures.py` re-tiles the run's example image, with pixels unchanged.
3. pdflatex, bibtex, pdflatex, pdflatex. Afterwards it prints the PDF page count, the last page of the
   main text (label `end-of-main-text`, placed after the Conclusion), undefined references and overfull
   boxes.

Build CPU is booked in `runs/paper_build_cpu_ledger.json` (cap 600 s).

## Numbers
Do not type numbers into `.tex` files. Every number comes from a macro in
`generated/numbers.tex`, and `generated/provenance.tsv` maps each macro to a source file, JSON field
and rule. Constants of the setting are the exception: vocabulary sizes, image size and chance-level
fractions are definitions, not results.

## Build history (ledger `runs/paper_build_cpu_ledger.json`, 13.3 CPU s of 600 used)
- `paper_build_v0`: pdflatex stopped with "Misplaced \noalign" (an `\input` inside `tabular`). The
  ledger status reads COMPLETED because the wrapper command ended with `echo`; no PDF was produced. The
  fix was to have the exporter write complete `tabular` environments.
- `paper_build_v0_retry` .. `paper_build_v0_3`: v0 builds. Fixed three overfull boxes (a wide
  equation, table widths), the title hyphenation and the figure size.
- `paper_build_v1` .. `paper_build_v1_4`: v1 builds with the paired and split results. Fixed two
  overfull boxes (table width, and an unbreakable run path in a caption, now `\url`).
- Final state: 13 pages; main text ends on page 8 (label `end-of-main-text`); no undefined
  references; no overfull boxes. Pages were rendered with `pdftoppm` and inspected visually
  (title block, tables 1-4, equation (1), bibliography, appendix).
