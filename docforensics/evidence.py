"""Evidence mapping (5.7): a zoomed crop for every region a finding points to, and a structured 'why / what to check' for every finding.

crops      each reference with a box is cut out of its page with some context, the box drawn in the finding's colour, and the crops of one
           finding are put side by side (a cross-document mismatch shows BOTH values). Aadhaar digits are blacked out as everywhere else.
guidance   a plain-language label (confirmed data conflict / potential alteration / observation), why the field or region needs a look, and
           the suggested check. Findings without a box (an unrecognised page, a missing field) get the text but no crop.
"""
import os
from PIL import Image, ImageDraw
from .dossier import redact_ids, COL, _font

LABELS = {"inconsistency": "Confirmed data conflict: two values in the submitted documents contradict each other (this says nothing about intent)",
          "potential_manipulation": "Potential alteration: an indicator that something may have been changed, not proof",
          "observed": "Observation: a note, coverage gap or weak signal, not an accusation"}


def guidance(f):
    """(label, why, check) for one Finding."""
    m, k = (f.method or "").lower(), f.kind
    label = LABELS.get(f.status, LABELS["observed"])
    if "arithmetic" in m:
        why = "Figures printed on the same page do not add up, so at least one of them is wrong or was changed after the document was produced."
        check = "Ask the issuer for the original or a recalculation and compare the figures with the payroll record."
    elif "income reconciliation" in m or "payer" in m or "case reference" in m or "gross income" in m or "annual" in m:
        why = "Records that should agree (payslip, bank statement, application, case reference) show different values."
        check = "Obtain the bank's own statement and the employer's payroll or confirmation record and see which side matches the issuer's data."
    elif k == "cross" and "coverage" not in m:
        why = "Documents that describe the same person or application disagree; at most one of them can be right."
        check = "Obtain the originals of the documents named, find which one matches the issuer's record, and do not assume which is correct."
    elif "pdf structure" in m or "pdf metadata" in m:
        why = "The way this PDF was built (fonts, drawing order, hidden layers, file history) differs from an untouched generated document."
        check = "Request the native or re-issued file from the issuer, inspect the region at full size and compare the two."
    elif k == "visual":
        why = "Pixels in this region behave differently from the rest of the page (compression, grain, tone, ink or typography). Edits can leave such traces, so can scanning."
        check = "Look at the region at full resolution on the original capture, repeat the capture if possible, and compare with the issuer's copy."
    elif k == "missing":
        why = "A check could not be completed or a value could not be read, so the absence of findings here is not evidence that the page is genuine."
        check = "Review the field or page manually, or resubmit a clearer capture."
    else:
        why = "A value breaks a rule that should hold within this document (format, date logic, balance or total)."
        check = "Verify the value against the original document and the issuing authority."
    return label, why, check


def potential_explanations(f):
    """Possible, non-exclusive explanations for a finding. They are hypotheses for the reviewer, never conclusions."""
    m, k, t = (f.method or "").lower(), f.kind, f.title.lower()
    if "arithmetic" in m or "balance" in m or "total" in m or "component" in m:
        return ["A typing or calculation error when the document was produced", "One figure edited after the document was produced", "A different version of the statement mixed with this one"]
    if "income reconciliation" in m or "payer" in m:
        return ["Timing: the payslip was corrected or reissued after the bank credit", "Net pay shown before an adjustment that the bank reflects", "A figure on the payslip or in the bank record was changed"]
    if "case reference" in m or "linkage" in m:
        return ["A document from another application was added by mistake", "A clerical error in the reference", "A reference altered to attach the document to this case"]
    if k == "cross":
        return ["A typing error in one of the documents", "The documents were produced at different times or are different versions", "A value in one document was changed", "A document of another person was mixed in"]
    if "signature" in m:
        return ["The copy is unsigned or signed electronically with a typed name", "The signature is faint or lost in compression", "A signature image added after the page was produced"]
    if "pdf structure" in m or "pdf metadata" in m:
        return ["A legitimate correction re-issued by the issuer", "A design element (badge, highlight) that looks like a patch", "A value overwritten after the document was generated"]
    if k == "visual":
        return ["Scanning, compression or lighting artefacts", "A value overwritten or pasted in after printing", "Different fonts or printers used legitimately on the same page"]
    if k == "missing":
        return ["Poor capture, faint print or an unusual layout", "The information is genuinely absent from the document"]
    if "duplicate" in m:
        return ["A second party or column (for example year-to-date) printed next to the first", "The page or row was copied or overwritten"]
    return ["An OCR or typing error", "The document does not follow the layout this check assumes", "The value was changed"]


def _crop(doc, ref, color):
    try:
        im = Image.open(doc.image_path).convert("RGB")
        redact_ids(doc, im)
        x0, y0, x1, y1 = [float(v) for v in ref["bbox"]]
        w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
        px, py = max(60.0, 0.6 * w), max(40.0, 1.6 * h)
        box = (max(0, int(x0 - px)), max(0, int(y0 - py)), min(im.width, int(x1 + px)), min(im.height, int(y1 + py)))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            return None
        ImageDraw.Draw(im).rectangle((x0, y0, x1, y1), outline=color, width=3)
        c = im.crop(box)
        s = max(1.0, min(3.0, 150.0 / c.height, 560.0 / c.width * 1.0)) if c.height < 150 else min(1.0, 560.0 / c.width)
        if abs(s - 1.0) > 0.01:
            c = c.resize((max(1, int(c.width * s)), max(1, int(c.height * s))), Image.LANCZOS)
        return c
    except Exception:
        return None


def _montage(crops, captions, color):
    """Crops side by side (two per row) with a caption line above each."""
    f = _font(13)
    cw = max(c.width for c in crops)
    cells = []
    for c, t in zip(crops, captions):
        cell = Image.new("RGB", (cw + 8, c.height + 24), "white")
        d = ImageDraw.Draw(cell)
        d.rectangle((0, 0, cell.width - 1, cell.height - 1), outline=(190, 190, 190))
        d.text((4, 4), t[:70], fill=color, font=f)
        cell.paste(c, (4, 20))
        cells.append(cell)
    rows = [cells[i:i + 2] for i in range(0, len(cells), 2)]
    W = max(sum(c.width for c in r) + 6 * (len(r) - 1) for r in rows)
    H = sum(max(c.height for c in r) for r in rows) + 6 * (len(rows) - 1)
    out = Image.new("RGB", (W, H), "white")
    y = 0
    for r in rows:
        x = 0
        for c in r:
            out.paste(c, (x, y))
            x += c.width + 6
        y += max(c.height for c in r) + 6
    return out


def build_evidence(docs, findings, outdir, max_crops=4):
    """{fid: {'label', 'why', 'check', 'crop': relative path or None, 'regions': n}} and the crop images under outdir/evidence/."""
    by_id = {d.doc_id: d for d in docs}
    os.makedirs(os.path.join(outdir, "evidence"), exist_ok=True)
    out = {}
    for f in findings:
        label, why, check = guidance(f)
        color = COL.get(f.status, (40, 110, 200))
        crops, caps, seen = [], [], set()
        for r in f.refs:
            d = by_id.get(r["doc_id"])
            key = (r["doc_id"], tuple(round(v) for v in r["bbox"])) if r.get("bbox") else None
            if d is None or not key or key in seen or len(crops) >= max_crops:
                continue
            seen.add(key)
            c = _crop(d, r, color)
            if c is not None:
                crops.append(c)
                caps.append(f"{r['doc_id']}" + (f" | {r['field']}" if r.get("field") else "") + (f" = {str(r['value'])[:24]}" if r.get("value") not in (None, "") and r.get("field") != "aadhaar_no" else ""))
        rel = None
        if crops:
            rel = os.path.join("evidence", f"{f.fid}.png")
            _montage(crops, caps, color).save(os.path.join(outdir, rel))
        out[f.fid] = {"label": label, "why": why, "check": check, "explanations": potential_explanations(f), "crop": rel.replace(os.sep, "/") if rel else None, "regions": len(crops)}
    return out
