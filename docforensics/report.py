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
