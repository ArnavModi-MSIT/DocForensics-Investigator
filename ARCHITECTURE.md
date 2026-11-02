# DocForensics: architecture, current behaviour and known gaps (review document)

**Purpose of this document.** It explains how the code works for the first four stages of the problem statement
(submission, classification, OCR/extraction, structural understanding), what is tested, what is not, and where the
gaps are, so reviewers can challenge the design. It is deliberately candid about weaknesses.

- Project: DataOps Innovation Hackathon 2026, Problem Statement 2 (AI-powered document forensics and contextual intelligence)
- Status date: 1 Nov 2026. Submission closes 7 Nov 2026, 11:59 PM IST. Deliverables: working prototype, PPT, demo video, GitHub repo.
- Size: about 2,400 lines of Python (largest: `extract.py` 511, `correlate.py` 184, `id_patterns.py` 172, `ingest.py` 169).
- Constraint we chose: a convincing, working demo pipeline first. Compute cost is not a constraint (heavier models are acceptable);
  every heavy component has a safe fallback so the demo cannot depend on network or API availability.

---

## 1. Problem statement in one paragraph

Build a system that analyses scanned/digital documents and **investigates** them rather than just reading them: extract
information, understand structure, find visual and contextual inconsistencies, correlate across related documents, and
produce an evidence-backed dossier with traceability to the source region. The required workflow starts with:

1. **Document Submission**: accept scanned documents, images and digital documents.
2. **Document Classification**: identify the type, determine structure and relevant fields.
3. **OCR and Information Extraction**: extract printed and handwritten information where applicable.
4. **Structural Understanding**: organise extracted information into structured representations while preserving the
   relationship with the original document.

Later stages (visual forensics, contextual verification, cross-document correlation, evidence, dossier) exist and are
summarised in section 7, but this document focuses on stages 1 to 4.

## 2. High-level architecture

```
                    +-----------------------------+
 files/zip  ------> | 1. ingest.py                |  expand zip safely, render PDF pages, hash source,
                    |    - deskew                 |  deskew, detect born-digital PDF pages
                    +--------------+--------------+
                                   | Document(page image, sha256, skew, born_digital)
                    +--------------v--------------+
                    | 3a. ocr.py (Tesseract)      |  Word(text, bbox, conf)
                    +--------------+--------------+
                                   | words -> visual lines
                    +--------------v--------------+
                    | 3b. extract.py              |  field extraction passes (fuzzy line, cells, id patterns)
                    |     + id_patterns.py        |  -> Field(key, raw, value, bbox, conf, match, label_sim)
                    +--------------+--------------+
                                   |
                    +--------------v--------------+      +------------------------+
                    | 2. classify() in extract.py |<---->| embed_classifier.py    |  bge-small embeddings,
                    |    candidate, veto, gate,   |      | (optional, cached)     |  prototypes incl. negatives
                    |    semantic label rescue,   |      +------------------------+
                    |    continuation pages       |
                    +--------------+--------------+
                                   | Document(doc_type, class_info, fields, facts)
              +--------------------+--------------------+
              |                                         |
   +----------v-----------+                  +----------v-----------+
   | forensics.py         |                  | correlate.py         |   4. relationships between fields:
   | (visual, per field)  |                  | intra + cross-doc    |      arithmetic, checksum, cross-doc
   +----------+-----------+                  +----------+-----------+
              +--------------------+--------------------+
                                   |  Finding(refs -> doc, field, value, bbox)
                    +--------------v--------------+
                    | dossier.py / audit.py / UI  |  PDF + JSON dossier, JSONL audit, Streamlit app
                    +-----------------------------+
```

Orchestration is in `pipeline.py::run_case`: ingest -> (per page) OCR, extract fields, classify -> continuation
inheritance -> forensics (per recognised raster page) -> intra/cross checks -> corroboration -> summary -> dossier.
Every step appends to an hash-chained audit log (`audit.py`).

### Core data model (`schema.py`)
| Object | Key attributes |
|---|---|
| `Word` | text, bbox (x0,y0,x1,y1), OCR confidence |
| `Field` | key (e.g. `dob`), label, raw text, normalised value, bbox of the **value**, OCR confidence, `match` (fuzzy / fuzzy-cell / semantic / pattern), `label_sim` |
| `Document` (one per page) | doc_id, source path, sha256, page image, doc_type, best_guess, type_conf (evidence score), skew_deg, born_digital, fields, `facts` (e.g. checksum result, forensics status), `class_info` (why the classification decision was made) |
| `Finding` | id, kind (visual / intra / cross / missing), severity, confidence, status (observed / inconsistency / potential_manipulation), explanation, `refs` (doc, field, value, bbox), metrics, method |

### Modules
| File | Role |
|---|---|
| `ingest.py` | zip expansion, PDF/image loading, deskew, born-digital detection |
| `ocr.py` | Tesseract wrapper, line grouping; auto-detects the Tesseract binary (`TESSERACT_CMD`, PATH, common Windows paths) |
| `extract.py` | labels dictionary, all extraction passes, classification, semantic label matching, continuation inheritance |
| `id_patterns.py` | label-free extractors for Indian IDs: Aadhaar (with Verhoeff checksum), PAN pattern, name above DOB, address to PIN code |
| `embed_classifier.py` | optional embedding backend (fastembed `BAAI/bge-small-en-v1.5`, else sentence-transformers), prototype texts per type |
| `forensics.py`, `correlate.py`, `dossier.py`, `llm.py`, `audit.py`, `debug.py`, `app.py` | downstream stages and UI |
| `synth/generate.py`, `tests/*` | synthetic benchmark, evaluation, calibration, ID-card test |

---

## 3. Stage 1: Document Submission (`ingest.py`)

**What it does**
- **Inputs:** JPG/PNG/TIFF/BMP/WEBP images, PDFs (scanned or generated), and ZIP bundles (one zip = one case).
- **PDFs** are rendered page by page with PyMuPDF at 200 DPI; each page becomes one `Document` (`<file>_p<n>`).
- **ZIP safety:** no `extractall`. Entries are read and written under a sanitised flat name. Rejected: path escapes (`../`, absolute),
  encrypted entries, unsupported types, files over 50 MB, more than 200 files or 300 MB total. Skips are logged to the audit trail
  and shown in the UI. Corrupt or unsupported single files are skipped, not fatal.
- **Provenance:** SHA-256 of each source file; the audit log stores file names (not local paths).
- **Deskew:** the page tilt is estimated by a projection-profile search (binarise at ~1000 px width, rotate over +/-8 deg in 0.5 deg
  steps then refine at 0.1 deg, pick the angle maximising row-profile sharpness). The page is rotated only when |angle| >= 0.3 deg
  and the sharpness gain is >= 3%, so straight digital pages are untouched. All later stages use the normalised page; the angle is logged.
- **Born-digital detection:** a PDF page is "born digital" if it has a text layer (>= 40 chars) and raster images cover < 35% of the page.
  Pixel-level forensics are skipped for such pages and the dossier says so.

**Tested:** synthetic images (digital, scan, phone-style), PDFs (generated and image-based), a hostile ZIP (path escape, absolute path,
corrupt member, junk files), and real files: a bank statement, an Aadhaar card, a sales report, a resume, a research paper.

**Gaps**
- TIFF reads only the first frame. HEIC, DOCX, XLSX are not accepted. Encrypted PDFs fail (skipped, logged).
- Deskew corrects rotation only. Perspective (keystone) distortion is not corrected (planned). Documents photographed on a busy background are not cropped.
- Page images are produced at a fixed 200 DPI; small text in dense documents may need a higher resolution.
- ZIP handling is now covered by automated tests (collisions, path escape, unsupported types).

---

## 4. Stage 2: Document Classification (`extract.py::classify`, `embed_classifier.py`)

**Supported types:** `application_form`, `id_card`, `salary_slip`, `bank_statement`. Anything else must come out as `unknown`.

**Design principle.** Keywords alone cannot separate a document *about* applications from an application form
(a research paper on a job app was classified as an application form in testing). So acceptance needs
*semantic* evidence, a *veto* against known unsupported classes, and *structural* evidence (real fields found).

**Decision procedure** (runs after fields are extracted, so structure can be used as evidence):
1. **Keyword evidence** `kw[type]`: share of the type's keywords present, matched as whole words/phrases (hyphen counts as part of a word, so
   "applicant-facing" does not match "applicant"). **Strong cues** (e.g. "Aadhaar", "UIDAI", "Government of India", "Income Tax Department")
   raise `kw[id_card]` to at least 0.8 only when **two** cues appear (cues now include Election Commission / Elector Photo Identity Card / driving licence wording), or one cue plus an Aadhaar/PAN pattern field was extracted (a Form 16 also says "Income Tax Department").
2. **Semantic evidence** (if the embedding model is available): the page text (first 2,000 characters of OCR) is embedded and compared by cosine
   with prototype descriptions: 2 or more per supported type, plus **negative classes** (`other:research_paper`, `other:resume`,
   `other:invoice`, `other:article_letter`). Candidate type = argmax of `similarity + 0.2 * keyword_score` over supported types.
3. **Veto:** if a negative class beats the best supported class by more than 0.03 cosine **and** keyword evidence for the candidate is < 0.5, the page is rejected.
   Valid documents with strong keywords cannot be vetoed.
4. **Semantic label rescue:** if the page clearly resembles a supported type (supported beats every negative class by >= 0.03), fields missing from the type's
   expected set are searched with embedding-based label matching (section 5, pass 4).
5. **Structural gate:** the page is accepted only if at least 3 of the candidate type's expected fields were extracted, **or** evidence is strong:
   keyword >= 0.75, the embedding's own top supported type agrees, no negative class resembles it, and >= 1 field exists
   (without a model: keyword >= 0.75 and >= 2 fields).
6. **Multi-page continuation:** a page with no structure (e.g. statement transaction pages) inherits the type of the preceding recognised page of the **same file** if
   no other type has real keyword support (< 0.5), the embedding still prefers that type and does not resemble an unsupported class (margin 0.05).
   It is marked as a continuation and skips missing-field checks.
7. **Fallback without the model:** keyword >= 0.5 plus the same structural gate.

**Outputs:** `doc_type`, `best_guess` (the embedding's own top pick, for display), `type_conf` (evidence score = 0.5 * keyword + 0.5 * field coverage),
`class_info` (keyword scores, expected fields found per type, semantic similarities, and a human-readable `decision` reason). The Streamlit tab
*Classification & OCR debug* shows all of this per page.

**Semantic category with confidence (added 2 Nov 2026).** Every page, supported or not, gets a ranked category from the same embedding model, over a wider list than the four supported types: ID card, application form, salary slip, bank statement plus corporate financial report, business/sales report, resume, research paper, invoice/bill, contract/policy, tax form, utility bill, academic certificate, medical/lab report, presentation/brochure, letter/article. Confidence = softmax(cosine / T) over all classes (T = `DOCFOR_CAT_TEMP`, 0.03): a *relative* score showing how clearly the best class beats the others, not a calibrated probability (clear documents score ~0.95-1.0, ambiguous ones 0.3-0.7). At **file level** the pages of one source file vote: per-page probability vectors averaged, weighted by text amount and by how decisive each page is, and the result states how many pages agree ("Business / sales report, confidence 0.39; 8 of 15 pages agree"). This does not change which documents are analysed: `doc_type` stays `unknown` for unsupported pages (the field gate is unchanged); the category only *describes* them in the summary, finding C-001, the dossier table and the debug tab. A page that looks like a supported type but fails the field gate is labelled "not confirmed by fields". Without the embedding model no category is produced.

**Tunables** (environment variables): `DOCFOR_MIN_FIELDS` (3), `DOCFOR_STRONG_KW` (0.75), `DOCFOR_KW_WEIGHT` (0.2), `DOCFOR_NEG_MARGIN` (0.03).

**Tested**
- 22 synthetic cases (4 pages each): 4/4 correct in every case, with and without capture degradation.
- Real files: the bank statement (4 pages, via continuation) and the Aadhaar card (2 pages) are accepted; a 15-page sales report, a resume and a research paper are rejected.
- Failures found and fixed during testing: resume mislabelled as a paper (more prototypes), keyword false positives ("issue" inside "issues"), a research paper accepted as a form (structural gate),
  a real bank statement rejected (gate too strict for real layouts), a sales report producing stray fields (value guards).

**Gaps**
- **Only four types.** Cheques, PAN cards as a distinct type, utility bills, Form 16, passports and others are not modelled.
- **Prototype texts and thresholds were tuned on small sets** (synthetic + about five real files). Cosine scores from this model are compressed (unrelated text is about 0.52 to 0.56), so margins of a few hundredths matter; these have not been validated on a large real corpus.
- Only the first page text is classified from a single signal family (text). No layout or visual model is used: logos, colours and card shapes are ignored.
- Only English is understood. Hindi or multilingual documents depend on English labels appearing.
- **"Structure"** is derived from extracted fields only (section 5 and 6); there is no learned layout classifier.
- Continuation inheritance could mislabel an unrelated page appended after a recognised document in a mixed PDF; the safeguards reduce this but it is heuristic.
- The semantic model must be downloaded once (about 70 MB) and cached; without it the system runs on keywords plus structure, which is weaker.

---

## 5. Stage 3: OCR and Information Extraction (`ocr.py`, `extract.py`, `id_patterns.py`)

**OCR.** Tesseract (`--psm 11`, sparse text) via pytesseract on the normalised page. Words come with boxes and confidences. Words are grouped into **visual lines** by clustering
vertical centres (tolerance 0.6 of median word height), then, where needed, into **cells** by splitting lines at horizontal gaps larger than max(2.5 x text height, 45 px).

**OCR engine layer (added 2 Nov 2026).** Tesseract is the baseline. If a page reads poorly (mean confidence < `DOCFOR_OCR_RETRY_CONF`=0.60, or < 8 words) the page is re-read, cheapest first: (1) **RapidOCR** (optional; the `rapidocr` package + `onnxruntime`, any Python >= 3.8, PaddleOCR PP-OCRv6 ONNX models bundled, Apache-2.0, ~2 s/page on CPU; the legacy `rapidocr_onnxruntime` package, Python < 3.13, is used if that is what is installed; its text-line boxes are split into words in proportion to character count); (2) only if still poor or RapidOCR is absent, **Tesseract preprocessing variants** (2x upscale + CLAHE, adaptive threshold, min-RGB channel). Candidates are ranked by `_score` = confidence mass of *plausible* words (>= 3 letters with a vowel, or tokens with digits) minus 0.15 per junk token, so a variant that emits hundreds of garbage tokens cannot win on volume. RapidOCR drops spaces ("SexMale", "DateofBirth:", "GOVERNMENTOFINDIA"), so `_respace` re-splits camelCase joins on line-leading tokens, tokens that fuzzy-match a known multi-word label, and all-letter tokens exactly covered by 2+ words from our own label/keyword/cue vocabulary (real names never split). Engine failures are logged and never abort the pipeline. Measured (legacy engine) on a synthetic Voter-ID-style card with tricolour/guilloche background (`tests/busy_card.py`, difficulty 0.2-0.7): Tesseract alone mean confidence 0.24-0.36, page unclassified, 0-1 usable fields; with the engine layer 0.91-0.96, classified `id_card`, name/father's name/EPIC (+DOB up to difficulty 0.5). Preprocessing variants alone did **not** help on that background (they raised word count, not accuracy), which is why RapidOCR is tried first. Benchmark (22 cases) and real-file regressions unchanged.

**Extraction runs as ordered passes; each pass only fills keys that are still missing, and every field records which pass produced it (`match`).**

| Pass | `match` | How it works | Guards |
|---|---|---|---|
| 1. Line-leading fuzzy | `fuzzy` | The first 1-4 words of a line are compared (rapidfuzz ratio) with a label dictionary (names, DOB, ID number, address, employer, income, gross/deductions/net pay, account no, period, opening/closing balance, currency, branch, IFSC, account type ...). Single-word labels need >= 90, multi-word >= 84. Value = the rest of the line. | Labels with digits or unbalanced brackets are ignored (`033 months)`); a value starting with a colon-terminated token means the label continues; loosely typed keys (`period`, `pay_month`, `currency`, `ifsc`, `branch`, `account_type`) need a plausible value |
| 2. Cell-based | `fuzzy-cell` | Lines are split into cells; a label at the start of any cell (or a multi-word label mid-cell, e.g. `BALANCE FORWARD` after dates) takes the rest of the cell or the next cell as its value. Handles multi-column headers (`ACCOUNT NO.` after an address column) | Same type guards; single-word labels only at cell start; value must not overlap an existing field |
| 3. Label-free patterns | `pattern` | **Voter ID (EPIC)** number (3 letters + 7 digits, key `epic_no`, separate from PAN/Aadhaar so ID kinds are never compared); **PAN stacked layout** (date alone on a line = DOB, the Latin line above = father's name, the one above that = holder's name); **Aadhaar number:** three 4-digit groups (or one 12-digit token) not part of a longer run (16-digit VIDs excluded), Verhoeff checksum recorded in `doc.facts`. **PAN pattern** (5 letters, 4 digits, 1 letter). **Name:** the closest Latin-script, capitalised 2-4-word line above the DOB line. **Address:** text after an `Address` label, continuing line by line in the same column until a 6-digit PIN code; low-confidence tokens and 1-2 letter fragments (Hindi/QR bleed) are dropped | Aadhaar numbers are stored **masked** in `raw` (`XXXX XXXX 1234`), masked in fields and exports, and blacked out (first 8 digits) in evidence images; full digits stay in memory for matching |
| 4. Semantic labels | `semantic` | Candidate label text from cell starts is embedded and compared with all label phrases (dictionary + synonyms such as "Total Earnings", "Net Payable", "Balance brought forward"). Accepted only if the best key is one of the **missing expected** fields, cosine >= 0.70, lead over the second-best key >= 0.06 | Runs only if the page clearly resembles a supported type; value type guard; no overlap with existing fields |

**Value normalisation:** dates -> ISO, money -> float (strips Rs/commas), IDs upper-cased, IFSC extracted by pattern. **Type guards** (`_value_ok`): dates must parse, money must parse, `period`
needs a date or month+year, `pay_month` a month name or month/year, `name` mostly letters with a capital and no `@`, `address` a digit or comma and >= 8 chars, `employer` letters and no `@`, IDs a digit and fewer than 2 hyphens
(so report codes like `RPT-2026-1507` are not IDs).

**Tested**
- Synthetic: about 95-100% of fields correct (exact normalised match) across digital, scan, hard scan and rough phone captures (46 fields per held-out pair). Remaining errors are OCR slips on ID letters (e.g. `FGHIJ5678K` read as `FGHI5678K`).
- Calibration on the real embedding model: 28 of 29 held-out label paraphrases mapped to the right field; junk phrases are separated mainly by the margin plus type guards.
- Real: the bank statement (page 1 yields account number, type, branch, currency, IFSC, statement date, name, balance forward), the Aadhaar card (front: name, DOB, masked number; back: masked number, address).
- A 15-page sales report produces **zero** fields after the guards (it produced stray fields before them).

**Gaps (against the requirement "extract printed and handwritten information")**
- **Handwriting is not supported.** Tesseract is a printed-text engine; there is no handwriting model or vision-LLM fallback.
- **Tables are not extracted.** Statement transaction rows are only skipped or classified as continuation pages; there is no table detector, no row/column model, and therefore no running-balance validation yet.
- **OCR slips** on IDs, dates and amounts remain the main source of false content mismatches. Planned fix: re-read typed fields with character-restricted, upscaled OCR.
- **Hindi/multilingual text and vertical text** (e.g. Aadhaar issue/print dates along the card edge) are not read; Hindi lines add junk that we filter heuristically.
- **Label dictionary is hand-written** for four document types plus synonyms; unseen labels depend on the embedding rescue.
- Unlabelled names are found for the Aadhaar-style layout (line above DOB) and the stacked PAN layout. Other card layouts (driving licence, passport) are not handled specifically.
- Multi-line values other than addresses (e.g. multi-line employer names) are cut at the line end.
- Pass ordering is greedy; an early wrong match can block a later correct one for the same key.
- Tesseract confidence is used as field confidence; it is not calibrated.

---

## 6. Stage 4: Structural Understanding (`schema.py`, `correlate.py`, `dossier.py`)

**What "structure" means in this code.** Every page becomes a `Document` holding typed `Field` objects. Each field keeps its **location** (bbox of the value on the normalised page),
**confidence**, **how it was matched** (fuzzy / cell / pattern / semantic) and the similarity score, so every later finding can point back to the exact region. The dossier JSON and PDF
expose these fields, the page classification reason, deskew angle and whether forensics ran.

**Relationships checked between fields (`correlate.py`)**
- *Within a document:* payslip arithmetic (gross - deductions = net pay), PAN format, plausible age, issue date not before date of birth, Aadhaar Verhoeff checksum, missing required fields
  (per type; ID cards and continuation pages are not required to have all fields).
- *Across documents of one case:* values for name, DOB, ID number, Aadhaar number, address, employer, account number are grouped by agreement. If one document disagrees with a majority, a single
  "odd-one-out" finding names it; with no majority it reports a pair and says it cannot tell which is right. Values are compared by **per-field comparators** (`compare.py`) that return same / compatible / near / different / unverifiable: names tolerate initials, word order, honorifics, "Surname, First" and omitted middle names (a single-word name is `unverifiable`); addresses compare by PIN code plus word overlap (different PIN = different; no PIN and little text = `unverifiable`); employers ignore legal suffixes (Pvt/Private/Ltd/Limited ...) and accept a shorter trade name; masked account/Aadhaar numbers compare on their last digits; IDs and dates are exact, with a one-character difference reported as `near`. Only `near`/`different` raise findings (text near-misses are medium because they are usually OCR slips; ID/date near-misses stay high); a finding whose values have OCR confidence < 0.60 is lowered one severity level and says so; pairs that could not be compared are summarised in one low note (N-004) so silence is never mistaken for agreement. **Income reconciliation** runs over all payslips and statement pages: each payslip's net pay must match some bank salary credit; a declared income is flagged when it exceeds the largest amount supported by the documents, where payslips contradicted by the bank do not count as support (a forger can raise slip and declaration together). With payslip gross available the flag is high; with net-only evidence it is medium and names the gross-vs-net possibility.
  Different ID kinds are never compared against each other (Aadhaar number and PAN are separate keys).
- *Corroboration:* a content inconsistency on a field that also shows visual evidence is raised to high severity and the visual finding gains confidence.

**Evidence mapping:** each page is exported with boxes drawn around the regions cited by findings (one box per region, labels merged, most serious colour on top).

**Tested:** on the synthetic benchmark all content issues (name/ID/DOB mismatches, arithmetic breaks, income inflation) are found; a corrupted Aadhaar number is flagged by the checksum; clean cases produce no findings.

**Gaps**
- **No table or section hierarchy:** the representation is a flat list of fields per page. There is no notion of sections, repeating rows or key-value groups, and no "information hierarchy" model.
- **Missing / duplicated / misplaced information** is only detected as "required field not found". Duplicated fields (the same value in two places) and misplaced fields (a value under the wrong label) are not detected.
- Relationships across pages of the same document (e.g. statement balances carried between pages) are not modelled.
- Field relationships are hard-coded rules for four types; there is no generic schema language.
- Cross-document matching assumes the documents in a case belong to one person; mixed cases can create spurious "mismatch" findings.
- OCR slips can produce high-severity false mismatches (the explanation includes the similarity so a reviewer can see it is a near-match).

---

## 7. Downstream stages (for context only)

- **Visual forensics (`forensics.py`)**: for each field, ELA, grain noise, paper tone and ink darkness are measured on background pixels near the text and compared with fields of **similar text size on the same page**
  (leave-one-out robust z-score), after flattening page lighting. A field is flagged when two independent signal families agree or one signal is overwhelming. Needs >= 4 comparable fields; skipped for born-digital PDFs; the dossier states when it did not run.
- **Dossier / audit / UI:** PDF and JSON dossier (summary, findings with evidence references, extracted fields, annotated pages, capability matrix, audit trail), hash-chained JSONL audit log, Streamlit app with a classification/OCR debug tab.
- **Optional LLM summary:** only explains findings the pipeline produced; deterministic fallback.

## 8. Evaluation and what the numbers mean

`synth/generate.py` builds fictional document sets (4 pages per case) in several capture conditions (digital, mild scan, hard scan, rough phone photo with 5 deg tilt and JPEG 45) with injected tampering. `tests/eval_synth.py`
scores classification, field extraction, issue recall and false positives per condition. Latest results (real embedding model active):

| Capture | Clean: false positives | Naive forgery found | Skilled forgery found | Classification | Field extraction |
|---|---|---|---|---|---|
| Digital | 0 | 5/5 | 0/8 | 4/4 per case | ~96-100% |
| Mild scan | 0 | 5/5 | 0/8 | 4/4 | ~98-100% |
| Hard scan (2 unseen people) | 0 | 10/10 | 0/8 | 4/4 | 100% |
| Rough phone photo (2 unseen people) | 0 | 10/10 | 0/8 | 4/4 | ~94-98% |

- "Naive" = pasted value with a different font, paper tone and ink. "Skilled" = same font, ink and tone matched, all numbers consistent across documents, so only pixel traces could reveal it.
- **Caveats:** the benchmark is self-generated, small (about 24 cases), the forensic thresholds were derived from the same family of data, and the naive forgery deliberately changes ink darkness, which is the cue the detector learned. Treat the results as evidence that the pipeline works end to end, **not as real-world accuracy**.
- **Real-document checks** (not a benchmark and not in the repo): bank statement, Aadhaar card, sales report, resume, research paper behave as described above. We do not yet have real salary slips, application forms, PAN cards or any real tampered documents.
- A dedicated test (`tests/test_id_cards.py`) builds fictional Aadhaar-style cards (valid and corrupted number) and checks classification, extraction, masking and the checksum finding.

## 9. Cross-cutting risks and limitations

1. **No real labelled data.** All forensic thresholds and most prototypes are tuned on synthetic documents plus a handful of real files.
2. **Skilled forgeries are invisible to pixel forensics** after re-capture; only content checks, issuer verification and (for PDFs) structure/metadata checks can help.
3. **Generated PDFs get no pixel forensics.** PDF structure, metadata, font and edit-history checks are planned, not implemented.
4. **OCR dependence.** A single engine (Tesseract); errors propagate into false mismatches.
5. **Model dependency.** The semantic classifier needs a one-time model download; behaviour degrades gracefully but is weaker without it.
6. **Platform.** Developed on Linux and Windows (Tesseract path auto-detected). Behaviour on macOS is untested.
7. **Licensing.** PyMuPDF is AGPL-3.0 (acceptable for a prototype; review before commercial use).
8. **Privacy.** Real documents must stay out of the repo and demo. Aadhaar numbers are masked in fields, exports, evidence images and ELA heat-maps (first 8 digits blacked out); unredacted normalised page copies stay in the local git-ignored `output/` folder, and OCR text may contain other personal data on the user's machine. The LLM summary is opt-in per run because it sends finding text to an external API.

## 10. Roadmap (candidate next steps, in priority order)

| # | Item | Why |
|---|---|---|
| 1 | Test real salary slip, application form, PAN card | Real layouts broke earlier versions twice; biggest unknown |
| 2 | Re-read typed fields (IDs, dates, amounts) with character-restricted upscaled OCR | Removes most OCR-slip false alarms |
| 3 | Transaction-table extraction and running-balance check for statements | Strengthens stage 4 and gives generated PDFs a real content check |
| 4 | PDF structure and metadata checks (producer, edit dates, font changes, incremental updates) | Gives born-digital documents a forensic signal |
| 5 | JPEG-ghost (compression-history) check | May expose pasted values in files with preserved compression history; useless after re-capture |
| 6 | Handwriting via a local handwriting model (no API/LLM dependency - decided 2 Nov 2026) | Closes the main gap against the stated requirement |
| 7 | Perspective correction, Hindi OCR, vertical text | Real-world capture robustness |

## 11. Questions we would like reviewers to challenge

1. Is the acceptance logic (semantic evidence + veto + structural gate) sound, or would a trained layout-aware classifier be a better use of the remaining time?
2. Which is the better investment for the demo: table extraction with running-balance validation, or a handwriting fallback?
3. Are the benchmark claims presented honestly enough (naive vs skilled forgeries, synthetic data)?
4. Are there document types or layouts evaluators are likely to try that we have not covered?
5. Is peer-relative visual forensics a defensible approach given that it cannot catch skilled forgeries?

## 12. Reproducing the results

```
pip install -r requirements.txt
pip install -r requirements-optional.txt   # optional: RapidOCR engine
python -m docforensics.embed_classifier --check      # one-time model download into ./models
python synth/generate.py && python synth/generate.py --hard
python tests/eval_synth.py                           # all cases (use DOCFOR_EMBED=off to compare without the model)
python tests/test_id_cards.py
streamlit run app.py
```

## 13. External review log (2 Nov 2026)

Five independent reviews were received. We reproduced each claimed defect before acting. **Fixed in the P0 batch** (each has a regression test in `tests/test_units.py`):

| Finding | Status |
|---|---|
| Zero amount crashes income reconciliation | Fixed |
| First extraction pass stored invalid values (`dob = INVALID`) | Fixed: one validation step (`finalize_value`) for every pass; unreadable values are reported as "could not be read", not silently skipped; trailing OCR junk is cut and the correction recorded |
| ZIP entries `a/b.jpg` and `a__b.jpg` overwrote each other; same-stem inputs shared a document ID | Fixed: unique storage names and document IDs |
| EXIF orientation ignored | Fixed |
| Unescaped OCR text crashed the PDF | Fixed: all dynamic text escaped; a PDF failure no longer loses the run (JSON dossier saved first) |
| UI case ID used raw in a path | Fixed: sanitised |
| Tesseract located at import (traceback on machines without it) | Fixed: located lazily with a clear error; OCR timeout added |
| PAN format applied to every ID number | Fixed: only for PAN-like or PAN-labelled values |
| "Masked in every export" was false (evidence images and heat-maps kept the digits) | Fixed: first 8 digits blacked out in both |
| "Append-only" audit log was truncated each run | Fixed: hash-chained per run with a final digest; wording corrected |
| LLM summary sent document values to an API by default | Fixed: opt-in per run |
| Evaluation scripts had no assertions; scoring accepted any finding on the right field | Fixed: strict scoring by finding kind, pass/fail gates, nonzero exit code, `pytest` collects 27 tests |
| Strong keyword cue too strong | Fixed: two cues, or one cue plus an ID pattern |
| Input bounds | Partly fixed: image side capped at 5,000 px, PDFs at 60 pages, per-page OCR timeout |

**Open (planned, in priority order):** typed-field re-OCR; payslip-to-credit matching by month (currently any credit within 1%); deskew-before-forensics (forensics should use original pixels and map boxes back); pixel constants scaled by text height instead of the 1000x1300 page assumption; coverage report of eligible/tested/flagged fields; real salary slip, PAN and form layouts; PDF structure/metadata checks; running-balance check; vision-based handwriting fallback.

**Added 2 Nov (Indian ID layouts):** Voter ID (EPIC) and unlabelled PAN layouts, `father_name` field, bilingual-label tolerance (leading low-confidence OCR noise is dropped before label matching, never values), same-ID back-of-card continuation, and a texture rule so decorated card backgrounds are not judged by peer-relative pixel forensics (a genuine PAN specimen had been flagged before this rule). Tested with mock OCR lines and a fictional specimen; real Voter ID and PAN scans still need to be re-run by the team.
