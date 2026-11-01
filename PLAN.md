# Plan: PS2 Document Forensics (submission closes 7 Nov 2026, 11:59 PM IST)

Deliverables (ALL THREE or disqualified): (1) PPT -> Google Drive link, public view; (2) demo video -> Drive link, public view; (3) GitHub repo + README (features, setup, run).

## Status (1 Nov, evening)
Done: ingest (images/PDF/ZIP), deskew, OCR, column-aware + label-free extraction (Aadhaar/PAN), embedding classifier with structural gate,
visual forensics, intra + cross checks, dossier PDF/JSON, audit trail, Streamlit UI incl. debug tab, synthetic + held-out benchmarks.
Next: fix A (typed-field re-OCR), PDF structure/metadata check, table extraction + running-balance check, real salary slip / PAN / form tests,
then PPT, demo video (synthetic documents only), final README pass. Before submitting: make the repo public or add the organisers.

## Schedule
| Day | Goal |
|---|---|
| Sun 1 Nov | Core pipeline working end to end (done). Push repo to GitHub. |
| Mon 2 Nov | Hackathon starts 2 PM. Re-read official brief/judging weights on Unstop; map gaps. Add 2-3 more document types/layouts (e.g. address proof, cheque) and a PDF-input sample. |
| Tue 3 Nov | Robustness: real-scan style noise (rotation, blur, phone-photo), OCR deskew, tune thresholds on a harder synthetic set; report precision/recall honestly. |
| Wed 4 Nov | Advanced extras (pick 1-2): signature region check, copy-move (ORB), PDF metadata forensics, LLM investigation notes. Submission window opens (4 Nov). |
| Thu 5 Nov | UI polish: side-by-side evidence view, confidence display, reviewer accept/reject notes in audit trail. |
| Fri 6 Nov | Record demo video (script below). Build PPT. Final README pass. |
| Sat 7 Nov | Buffer, submit before 11:59 PM (aim for 6 PM). Verify the 3 links open in a private window. |

## Demo video script (4-5 min)
1. Problem (20s): OCR reads, it does not verify. 2. Upload the 4-document tampered case. 3. Progress + extracted fields with confidence.
4. Findings: visual indicator on Net Pay + arithmetic break + bank mismatch (corroboration). 5. Evidence map and ELA heat-map, trace finding to region.
6. Clean case -> zero findings. 7. Dossier PDF + audit trail. 8. Capability matrix and honest limitations.

## PPT outline (8-10 slides)
Problem -> Approach/architecture -> Visual forensics (peer-relative idea) -> Cross-document reasoning -> Evidence & dossier -> Demo screenshots -> Results (clearly labelled synthetic) -> Capability matrix/limits -> Roadmap.
