"""make_final.py -- build the four JAGST submission files (Revision 1, manuscript 1283).

Run from ms/ with FIGDIR=fig_final so the light figures are embedded.
Outputs in out/docs/final/:
  JAGST_1283_Revised_Manuscript_Clean.docx
  JAGST_1283_Revised_Manuscript_Tracked_Changes.docx
  JAGST_1283_Supplementary_Material.docx
  JAGST_1283_Cover_Letter_and_Response_to_Reviewer.docx
"""
import os
import re
import shutil
import subprocess
import zipfile

import build

RT = 'Spectral soil nutrient prediction under spatial transfer'
FINAL = os.path.join(build.DOCS, 'final')
os.makedirs(FINAL, exist_ok=True)
NAMES = {
    'clean': 'JAGST_1283_Revised_Manuscript_Clean.docx',
    'tracked': 'JAGST_1283_Revised_Manuscript_Tracked_Changes.docx',
    'supp': 'JAGST_1283_Supplementary_Material.docx',
    'letter': 'JAGST_1283_Cover_Letter_and_Response_to_Reviewer.docx',
}
P = {k: os.path.join(FINAL, v) for k, v in NAMES.items()}

# 1. clean manuscript
text = build.render(['part0_front.md', 'part1_intro_methods.md', 'part2_results.md', 'part3_end.md'],
                    P['clean'], line_numbers=False)
build.finalise(P['clean'], RT)
missing, refset = build.check_citations(text)
body = text.split('**8.0 References**')[0]
abst = text.split('**Abstract**')[1].split('**Keywords')[0]
print('citations without reference:', missing)
print('abstract words:', len(abst.split()))

# 2. supplementary material (landscape) and cover letter
build.render(['supp.md'], P['supp'], line_numbers=False, landscape=True)
build.finalise(P['supp'], RT, refs_heading=None)
build.render(['letter.md'], P['letter'], line_numbers=False)
build.finalise(P['letter'], None, refs_heading=None)

# 3. tracked changes: round-1 manuscript -> clean revision
tmp = os.path.join(FINAL, 'tracked_tmp.docx')
subprocess.run(['python3', 'compare.py', os.path.join(build.DOCS, 'original_round1_light_anon.docx'),
                P['clean'], tmp], check=True)
with zipfile.ZipFile(tmp) as zin, zipfile.ZipFile(P['tracked'], 'w', zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename in ('word/document.xml', 'docProps/core.xml', 'docProps/app.xml'):
            s = data.decode('utf8')
            s = s.replace('w:author="Unknown Author"', 'w:author="Revision 1"')
            s = re.sub(r'<dc:creator>.*?</dc:creator>', '<dc:creator></dc:creator>', s)
            s = re.sub(r'<cp:lastModifiedBy>.*?</cp:lastModifiedBy>', '<cp:lastModifiedBy></cp:lastModifiedBy>', s)
            data = s.encode('utf8')
        zout.writestr(item, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
os.remove(tmp)
for k, p in P.items():
    print(f'{os.path.basename(p):60s} {os.path.getsize(p) / 1024:8.0f} KB')
