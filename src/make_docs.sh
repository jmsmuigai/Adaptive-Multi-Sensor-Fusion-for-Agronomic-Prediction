#!/bin/bash
# Regenerate every table, figure, number and Word document from stored results.
set -e
cd "$(dirname "$0")"
python3 results.py && python3 make_numbers.py > out/numbers_preview.txt
python3 fig_study.py && python3 fig_arch.py && python3 fig_results.py
python3 tables_ms.py
python3 verify.py
cd ms
python3 build.py
python3 - <<'PY'
import build, os
from docx import Document
from docx.shared import Pt
build.render(['letter.md'], os.path.join(build.DOCS, 'response_to_reviewer.docx'), line_numbers=False)
build.render(['supp.md'], os.path.join(build.DOCS, 'supplementary_material.docx'), line_numbers=False, landscape=True)
f = os.path.join(build.DOCS, 'response_to_reviewer.docx'); d = Document(f)
for p in d.paragraphs:
    p.paragraph_format.line_spacing = 1.15; p.paragraph_format.space_after = Pt(6)
d.save(f)
PY
python3 compare.py ../out/docs/original_round1.docx ../out/docs/manuscript_clean.docx ../out/docs/manuscript_tracked.docx
cd ../out/docs && rm -rf trx && mkdir trx && cd trx && unzip -q ../manuscript_tracked.docx \
  && sed -i 's/w:author="Unknown Author"/w:author="Revision 1"/g' word/document.xml \
  && rm ../manuscript_tracked.docx && zip -qXr ../manuscript_tracked.docx . && cd .. && rm -rf trx
echo ALL DOCUMENTS BUILT
