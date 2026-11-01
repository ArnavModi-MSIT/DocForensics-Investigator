# DocForensics - AI-Powered Document Forensics & Contextual Intelligence

DataOps Innovation Hackathon 2026 (Problem Statement 2). Goes beyond OCR: it extracts fields, tests each
field for signs of visual tampering, checks documents against themselves and against each other, and
produces an evidence-mapped investigation dossier (PDF + JSON) with a full audit trail.

> Visual findings are **indicators** of possible alteration, not proof of fraud. All sample data is fictional.

## Pipeline
`Submission -> Classification -> OCR & extraction -> Visual forensics -> Contextual verification -> Cross-document correlation -> Evidence mapping -> Dossier`

| Stage | Module | How |
|---|---|---|
| Ingest (images/PDF, multi-doc) | `ingest.py` | PyMuPDF render, SHA-256 per source, page IDs |
| OCR + fields | `ocr.py`, `extract.py` | Tesseract word boxes, line grouping, fuzzy label->field map, per-field confidence + bbox |
| Classification | `extract.py` | keyword scoring (id card, salary slip, bank statement, application form) |
| Visual forensics | `forensics.py` | per-field ELA, paper-grain noise, paper tone, ink darkness, compared with the **same page's other fields** (leave-one-out robust z-score). Flags need agreement of independent signal families or one overwhelming signal |
| Contextual checks | `correlate.py` | payslip arithmetic, ID format, date logic, plausibility, completeness |
| Cross-document | `correlate.py` | entity matching (exact + fuzzy), odd-one-out detection, income reconciliation (declared vs payslip vs bank) |
| Corroboration | `correlate.py` | content inconsistency on a field that also shows visual tampering -> severity/confidence raised |
| Reasoning/summary | `llm.py` | optional Claude summary grounded only in produced findings; template fallback |
| Dossier + audit | `dossier.py`, `audit.py` | annotated pages, PDF/JSON, append-only JSONL audit log |

Every finding carries `refs` = (document, field, value, bbox) so it traces back to the exact region of the source.
Findings are labelled `potential_manipulation` (visual indicator), `inconsistency` (content conflict) or `observed`.

## Setup & run
Requires Python 3.10+ and the Tesseract binary (`apt install tesseract-ocr` / Windows installer from UB-Mannheim).
```
pip install -r requirements.txt
python synth/generate.py                      # fictional sample cases -> samples/
python -m docforensics.pipeline samples/case_tampered --out output/case_tampered
python tests/eval_synth.py                    # recall / false-positive check on synthetic ground truth
streamlit run app.py                          # UI
```
Optional LLM summary: `set ANTHROPIC_API_KEY=...` (model via `DOCFOR_MODEL`). Without it a deterministic summary is used.
Windows: if Tesseract is not on PATH set `pytesseract.pytesseract.tesseract_cmd` or add it to PATH.

## Results on bundled synthetic data (small, self-generated - treat as a smoke test, not a benchmark)
- `case_clean`: 0 findings. `case_tampered`: 5/5 injected issues found (2 pasted-over values, name/ID mismatches, inflated income) plus corroborated arithmetic/income conflicts.

## Capability matrix
| Capability | Status |
|---|---|
| Images + PDF ingestion, multi-document cases | Implemented |
| Printed-text OCR, label:value fields, confidence, bbox | Implemented (Tesseract) |
| Document classification | Implemented (keywords); ML layout classification planned |
| ELA / noise / tone / ink-darkness tamper indicators | Implemented (peer-relative) |
| Arithmetic, format, date, completeness checks | Implemented |
| Cross-doc entity matching, income reconciliation | Implemented |
| Evidence maps, dossier (PDF/JSON), audit trail, UI | Implemented |
| Handwriting recognition | Unsupported |
| Signature analysis, copy-move detection, PDF metadata forensics | Planned |
| Layout-aware models (LayoutLM), multilingual OCR | Planned |

## Known limitations
- Needs >= 4 extracted fields per page for peer comparison; sensitive to heavy re-compression / re-scanning of the whole page.
- Template-style `Label: value` layouts; free-form or tabular documents need extra extraction rules.
- OCR errors can create false content mismatches (fuzzy-match similarity is reported so reviewers can judge).
- Thresholds were tuned on synthetic data; validate on real documents before any production use.
- Licences: Tesseract (Apache-2.0), OpenCV (Apache-2.0), PyMuPDF (AGPL-3.0 - fine for this prototype, review before commercial use), others permissive.
