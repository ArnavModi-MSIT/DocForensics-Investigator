"""Dossier content that is specific to one case (5.8): confidence labels, what was checked, and the areas that still need verification.

Nothing here adds findings. It summarises what ran, what could not run and what could not be compared, so a reviewer sees the limits of THIS case
and not only the generic limitations of the method.
"""
import os

CONF_SCALE = ("Confidence levels: high (0.80 and above), medium (0.60 to 0.79), low (below 0.60). The number is the system's own estimate of how reliable the "
              "specific check is on this evidence (for example an exact arithmetic mismatch scores high, a weak pixel signal scores low). It is not a probability "
              "that a document is forged, and it does not say which of two conflicting documents is correct.")
GROUPS = [("inconsistencies", "Identified inconsistencies within a document", ("intra",)),
          ("visual", "Visual forensic observations", ("visual",)),
          ("cross", "Cross-document discrepancies", ("cross",))]


def confidence_level(c):
    return "high" if c >= 0.8 else ("medium" if c >= 0.6 else "low")


def group_of(f):
    """inconsistencies | visual | cross | notes (coverage and data-quality notes go to the verification section)."""
    if f.kind == "missing":
        return "notes"
    return {"intra": "inconsistencies", "visual": "visual", "cross": "cross"}.get(f.kind, "inconsistencies")


def processing_summary(docs, findings, corr):
    """Short facts about what ran: pages, how the text was read, fields, tables, checks."""
    rec = [d for d in docs if d.doc_type != "unknown"]
    layer = [d for d in docs if d.facts.get("ocr_engine") == "pdf-text-layer"]
    ocr = [d for d in docs if d.facts.get("ocr_engine") not in (None, "pdf-text-layer")]
    nf = sum(len(d.fields) for d in docs)
    tabs = sum(len(d.tables) for d in docs)
    m = (corr or {}).get("matrix", [])
    cnt = lambda v: sum(1 for x in m if x["verdict"] == v)
    pix = [d for d in rec if d.facts.get("forensics") == "run"]
    pdf = [d for d in rec if d.born_digital]
    lines = [f"{len(docs)} page(s) from {len({d.path for d in docs})} file(s); {len(rec)} recognised as a supported type, {len(docs) - len(rec)} not recognised.",
             f"Text: {len(layer)} page(s) read exactly from the PDF text layer, {len(ocr)} page(s) read by OCR.",
             f"{nf} field(s) extracted, {tabs} table(s) found.",
             f"Visual forensics: pixel-level checks on {len(pix)} page(s), PDF structure checks on {len(pdf)} generated PDF page(s).",
             f"Cross-document: {len(m)} field(s) compared across documents ({cnt('agree')} agree, {cnt('differ')} differ, {cnt('single')} in one document only, "
             f"{cnt('unverifiable')} could not be compared); {len((corr or {}).get('links', []))} document pair(s) linked by shared identifiers.",
             f"{len(findings)} finding(s) in total."]
    return lines


def verification_areas(docs, findings, corr):
    """Case-specific list of what a reviewer should still verify: [{'area', 'detail'}]."""
    out = []
    add = lambda area, detail: out.append({"area": area, "detail": detail})
    for d in docs:
        if d.doc_type == "unknown":
            add(f"{d.doc_id}: page not recognised", f"No field-level or forensic checks ran on it. {d.class_info.get('decision', '')}".strip())
    for d in docs:
        conf = d.facts.get("ocr_mean_conf")
        if d.doc_type != "unknown" and conf is not None and conf < 0.6:
            add(f"{d.doc_id}: low read confidence ({conf})", "Values may be misread; a misread value can create or hide a mismatch. Check the fields against the original.")
    from .extract import EXPECTED
    for d in docs:
        done = set()
        for u in d.facts.get("unparsed", []):
            if d.doc_type != "unknown" and not d.get(u["key"]) and u["key"] in EXPECTED.get(d.doc_type, ()) and u["key"] not in done:
                done.add(u["key"])
                add(f"{d.doc_id}: '{u['key']}' could not be read", f"The label was found but the value ({str(u['raw'])[:30]!r}) is not valid for this field; checks that need it were skipped.")
    seen = set()
    for f in findings:
        if f.kind == "missing" and f.fid[0] in "CN" and f.method != "input type detection" and f.title not in seen:
            seen.add(f.title)
            add(f.title, f.explanation[:260])
    pdf = [d for d in docs if d.born_digital and d.doc_type != "unknown"]
    if pdf:
        add(f"{len(pdf)} generated PDF page(s): pixel analysis does not apply",
            "Only file-structure checks were possible. A document altered before it was turned into a PDF, or issued with wrong values, passes them. Compare with the issuer's own copy.")
    nopix = [d for d in docs if d.doc_type != "unknown" and not d.born_digital and str(d.facts.get("forensics", "")).startswith("not run")]
    if nopix:
        add(f"{len(nopix)} page(s): visual forensics could not run", "Too few comparable fields; the absence of visual findings there is not evidence of authenticity.")
    for m in (corr or {}).get("matrix", []):
        if m["verdict"] == "unverifiable":
            add(f"{m['field']}: could not be compared", "The values were too short or incomplete to compare reliably; no mismatch was raised and none was ruled out.")
    single = [m["field"] for m in (corr or {}).get("matrix", []) if m["verdict"] == "single"]
    if single:
        add("Fields present in only one document", ", ".join(single) + ". They could not be cross-checked; confirm them against an independent source.")
    for did in (corr or {}).get("unlinked", []):
        add(f"{did}: shares no identifier with the other documents", "Confirm that it belongs to this application.")
    for d in docs:
        if str(d.facts.get("signature", "")).startswith("0 typed") or any(not r.get("found", True) for r in d.facts.get("regions", []) if r.get("kind") == "signature"):
            add(f"{d.doc_id}: no signature mark found next to a signature label", "The document may be unsigned or the signature too faint to detect.")
    add("Issuer verification", "No document was checked against the issuing organisation or a registry. Confirm identity, employment and account details with the issuers.")
    add("Original documents", "Findings come from the submitted copies only. Compare every flagged value and region with the originals before any decision.")
    return out


# ---- 7.1 / 7.3 / 7.4 / 7.6 helpers ---------------------------------------------------------------------------------------------
def source_info(d, all_docs):
    """Where a page came from: file name and type, size, page number of the file, page size in pixels."""
    n_pages = sum(1 for x in all_docs if x.path == d.path)
    pg = [x.doc_id for x in all_docs if x.path == d.path].index(d.doc_id) + 1
    try:
        size = os.path.getsize(d.path)
    except OSError:
        size = None
    ext = os.path.splitext(d.path)[1].lstrip(".").upper() or "?"
    return {"file": os.path.basename(d.path), "file_type": ext, "file_bytes": size, "page": pg, "pages_in_file": n_pages, "page_px": [d.width, d.height]}


def processing_status(d):
    """One line per page: how the text was read, whether it was recognised, which forensic checks ran."""
    eng = d.facts.get("ocr_engine")
    read = "text read exactly from the PDF text layer" if eng == "pdf-text-layer" else f"read by OCR ({eng}, mean confidence {d.facts.get('ocr_mean_conf')})"
    rec = (f"recognised as {d.doc_type.replace('_', ' ')}" + (f" (continuation of {d.class_info['continuation_of']})" if d.class_info.get("continuation_of") else "")
           if d.doc_type != "unknown" else "not recognised as a supported type")
    if d.doc_type == "unknown":
        checks = "no field or forensic checks ran"
    elif d.born_digital:
        checks = "PDF structure checks ran; pixel analysis not applicable"
    elif d.facts.get("forensics") == "run":
        checks = "pixel-level forensics ran"
    else:
        checks = "pixel-level forensics could not run (" + str(d.facts.get("forensics", "")).replace("not run: ", "")[:70] + ")"
    return f"{rec}; {read}; {len(d.fields)} field(s); {checks}"


MISSING_M = ("schema completeness", "value validation")
DUP_M = ("duplicate field check", "duplicate table row check", "duplicate reference check", "duplicate page check")
MISPLACED_M = ("value-kind check", "section placement check")


def contextual_lists(findings):
    """Findings about missing, duplicated and misplaced information, as dicts {fid, title, detail, refs}."""
    pick = lambda ms: [{"fid": f.fid, "title": f.title, "severity": f.severity, "detail": f.explanation[:240]} for f in findings if f.method in ms]
    return {"missing": pick(MISSING_M), "duplicated": pick(DUP_M), "misplaced": pick(MISPLACED_M)}


VISUAL_GROUPS = [("regions", "Suspicious regions and potential alterations"), ("font_structure", "Font, alignment and structural inconsistencies"),
                 ("image", "Image manipulation observations"), ("signature", "Signature-related observations")]


def visual_group_of(f):
    m = (f.method or "").lower()
    if "signature" in m:
        return "signature"
    if any(k in m for k in ("typography", "line spacing", "column alignment", "pdf structure", "pdf metadata")):
        return "font_structure"
    return "image"


def visual_bucket(f):
    """Each visual finding is shown once: signature observations; other possible alterations; font/alignment/structure; image manipulation."""
    if visual_group_of(f) == "signature":
        return "signature"
    if f.status == "potential_manipulation":
        return "regions"
    return visual_group_of(f)


def visual_groups(docs, findings):
    """The visual findings split by kind of observation, each with what was checked (so 'nothing found' is stated, not implied)."""
    vis = [f for f in findings if f.kind == "visual"]
    rec = [d for d in docs if d.doc_type != "unknown"]
    checked = {"font_structure": [], "image": [], "signature": []}
    for d in rec:
        if d.born_digital:
            checked["font_structure"].append(f"{d.doc_id}: PDF structure checks ({(d.facts.get('pdf_checks') or {}).get('amount_fonts', 'run')})")
            checked["image"].append(f"{d.doc_id}: not applicable (generated PDF, no scan or photo)")
        else:
            checked["font_structure"].append(f"{d.doc_id}: typography {d.facts.get('typography', 'not run')}")
            checked["image"].append(f"{d.doc_id}: pixel forensics {d.facts.get('forensics', 'not run')}; JPEG grid {d.facts.get('grid', 'not run')}; photo {d.facts.get('photo', 'not run')}")
        checked["signature"].append(f"{d.doc_id}: {d.facts.get('signature', 'not run')}")
    return [{"key": key, "title": title, "findings": [f.fid for f in vis if visual_bucket(f) == key], "checked": checked.get(key, [])} for key, title in VISUAL_GROUPS]


def anomaly_rows(findings, docs=None):
    """The consolidated anomaly table for the summary: every finding about the documents (coverage notes excluded)."""
    rows = []
    for f in sorted([f for f in findings if f.kind != "missing"], key=lambda x: ({"high": 0, "medium": 1, "low": 2}[x.severity], -x.confidence)):
        rows.append({"fid": f.fid, "title": f.title, "severity": f.severity, "status": f.status, "confidence": f"{confidence_level(f.confidence)} ({f.confidence:.2f})",
                     "documents": sorted({r["doc_id"] for r in f.refs})})
    return rows


# ---- standards and reference frameworks (informational; no compliance claim) -------------------------------------------------------
STANDARDS_NOTE = ("This system draws on the frameworks below as reference points. It does not claim compliance or certification with any of them; "
                  "each row says what we do and what we do not do.")
STANDARDS = [
    ("ISO/IEC 27037 (identification, collection, acquisition and preservation of digital evidence)",
     "Each source file is hashed (SHA-256) at ingest; originals are never modified (analysis uses derived copies, recorded in the log); every step is written to a hash-chained audit log with archived run history.",
     "No acquisition or custody control before upload: the system only sees the copy it is given. No write protection or collection records for the original device or system."),
    ("ISO/IEC 27041 (suitability and adequacy of investigative methods)",
     "Methods are documented (ARCHITECTURE.md), tested on a synthetic benchmark and a separately generated synthetic test pack with an answer key, and limitations are stated openly (skilled forgeries, tuning on synthetic data).",
     "No formal validation study, no measured error rates on real documents, no independent testing."),
    ("ISO/IEC 27042 (analysis and interpretation of digital evidence)",
     "Findings separate a data inconsistency, a potential alteration and an observation; each has a confidence level, possible explanations and a suggested check; nothing is called fraud; the pipeline is deterministic and repeatable.",
     "Confidence levels are the system's own estimates, not calibrated probabilities; no formal competence or review process around the analyst."),
    ("NIST SP 800-86 (forensic techniques in incident response)",
     "The pipeline follows its collection, examination, analysis and reporting phases: ingest and hashing; reading, extraction, structure and forensic checks; contextual and cross-document analysis; the dossier.",
     "It is written for incident response on live systems; volatile data, network and system sources are out of scope here."),
    ("OWASP Machine Learning Security Top 10",
     "No training is done and no model is hosted, so model theft, poisoning, inversion and membership inference do not apply. Supply chain: the downloaded embedding model is checked against recorded SHA-256 values and the result is logged. Output integrity: hash-chained audit and review logs. The optional Claude summary is off by default.",
     "Input manipulation (documents crafted to mislead OCR or classification) is only partly mitigated (veto rule, minimum expected fields) and not tested adversarially. Recorded model hashes were taken from our own first download, not from a publisher signature. RapidOCR models are bundled in their package and not separately pinned. When the optional summary is on, finding text (document values) is sent to an external API."),
]
