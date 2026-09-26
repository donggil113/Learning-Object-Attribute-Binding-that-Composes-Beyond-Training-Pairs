#!/usr/bin/env bash
# Build the working draft: export numbers from raw runs, compile, report page facts.
# Run from the repository root under the build ledger:
#   python3 scripts/ledger_run.py --name paper_build --ledger runs/paper_build_cpu_ledger.json \
#       --cap-s 600 -- bash paper/build.sh
set -u
python3 scripts/export_paper_numbers.py || exit 1
.venv/bin/python scripts/make_paper_figures.py || exit 1
cd paper
rm -f main.aux main.bbl main.blg main.out
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build_pass1.log 2>&1 || { tail -20 main.log; exit 1; }
bibtex main > build_bibtex.log 2>&1 || { cat build_bibtex.log; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build_pass2.log 2>&1 || exit 1
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build_pass3.log 2>&1 || exit 1
python3 - <<'EOF'
import re
aux = open("main.aux").read()
log = open("main.log", errors="ignore").read()
end = re.search(r"\\newlabel\{end-of-main-text\}\{\{[^}]*\}\{(\d+)\}", aux)
refs_start = re.search(r"\\newlabel\{app:chance\}\{\{[^}]*\}\{(\d+)\}", aux)
pages = re.search(r"Output written on main.pdf \((\d+) pages", log)
undefined = re.findall(r"(?:Citation|Reference) `([^']+)' on page \d+ undefined", log)
overfull = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)
print(f"pdf_pages={pages.group(1) if pages else None} main_text_last_page={end.group(1) if end else None} "
      f"appendix_first_page={refs_start.group(1) if refs_start else None} "
      f"undefined={sorted(set(undefined))} overfull_pt={overfull}")
EOF
