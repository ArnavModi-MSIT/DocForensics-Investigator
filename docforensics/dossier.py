"""Investigation dossier (5.8): JSON + PDF with annotated evidence."""
import json, os
from PIL import Image, ImageDraw, ImageFont

COL = {"potential_manipulation": (220, 30, 30), "inconsistency": (240, 140, 0), "observed": (40, 110, 200)}
CAPABILITIES = [
    ("Ingestion: images, PDFs and ZIP bundles (one zip = one case); multi-document cases", "Implemented"),
    ("Page normalisation: deskew (tilt up to +/-8 deg) and lighting flattening for phone photos", "Implemented; perspective correction planned"),
    ("OCR + field extraction with field-level confidence and bounding boxes", "Implemented (Tesseract + optional RapidOCR; printed text, label:value layouts)"),
    ("Semantic label matching (embeddings) for field labels outside the dictionary", "Implemented (needs embedding model; falls back to fuzzy dictionary)"),
    ("Document classification: embeddings + keywords, unsupported-class veto, structural gate (>=3 expected fields)",
     "Implemented (keyword + structure fallback when no model)"),
    ("Visual forensics: ELA, grain/noise, paper tone, ink darkness (peer-relative, lighting-corrected)", "Implemented"),
    ("Contextual checks: arithmetic, formats, date logic, completeness", "Implemented"),
    ("Cross-document entity matching (odd-one-out) and income reconciliation", "Implemented"),
    ("Evidence mapping (annotated regions), audit trail, reviewer UI, PDF/JSON dossier", "Implemented"),
    ("Handwriting recognition", "Unsupported"),
    ("Signature analysis, copy-move detection, PDF metadata forensics", "Planned"),
    ("Layout-aware ML models (LayoutLM etc.), table extraction, multilingual OCR", "Planned"),
]


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def redact_ids(doc, im):
    """Black out the first 8 of 12 digits of any detected Aadhaar number in an evidence image (PIL image, in place).
    Evidence images are embedded in the PDF and shown in the UI, so they must not leak the number."""
    d = ImageDraw.Draw(im)
    for f in doc.fields:
        if f.key == "aadhaar_no":
            x0, y0, x1, y1 = f.bbox
            d.rectangle((x0 - 4, y0 - 4, x0 + int((x1 - x0) * 0.68), y1 + 4), fill=(0, 0, 0))


def annotate(doc, findings, out_path):
    """Draw one box per region; labels of all findings on that region are joined, and the
    most serious status (potential manipulation) sets the colour and is drawn last (on top)."""
    rank = {"observed": 0, "inconsistency": 1, "potential_manipulation": 2}
    im = Image.open(doc.image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    f = _font(20)
    regions = {}
    for fd in findings:
        for r in fd.refs:
            if r["doc_id"] == doc.doc_id and r.get("bbox"):
                regions.setdefault(tuple(r["bbox"]), []).append(fd)
    for bbox, fds in sorted(regions.items(), key=lambda kv: max(rank[x.status] for x in kv[1])):
        top = max(fds, key=lambda x: rank[x.status])
        c = COL.get(top.status, (0, 0, 0))
        x0, y0, x1, y1 = bbox
        d.rectangle((x0 - 6, y0 - 6, x1 + 6, y1 + 6), outline=c, width=3)
        d.text((x0 - 6, y0 - 30), ", ".join(sorted({x.fid for x in fds})), fill=c, font=f)
    redact_ids(doc, im)
    im.save(out_path)
    return out_path


def _cat_cell(d):
    """Page-level category with its confidence; for unsupported pages also the file-level call."""
    c = d.get("category_page")
    if not c:
        return "-"
    t = f"{c['label']} {c['confidence']:.2f}"
    if d["type"] == "unknown" and not str(c.get("key", "")).startswith("other:"):
        t += " - not confirmed by fields"                 # looks like a supported type but too few expected fields were found
    f = d.get("category_file")
    if d["type"] == "unknown" and f and f["label"] != c["label"]:
        t += f" (file: {f['label']} {f['confidence']:.2f})"
    return t


def build_json(case_id, docs, findings, summary, summary_src, audit_events, audit_digest=""):
    return {
        "case_id": case_id, "audit_digest": audit_digest, "summary": summary, "summary_source": summary_src,
        "documents": [{"doc_id": d.doc_id, "type": d.doc_type, "type_conf": d.type_conf, "sha256": d.sha256,
                       "size": [d.width, d.height], "decision": d.class_info.get("decision", ""),
                       "best_guess": d.best_guess, "category_page": d.class_info.get("category"), "category_file": d.class_info.get("file_category"),
                       "skew_deg": d.skew_deg, "born_digital": d.born_digital, "forensics": d.facts.get("forensics", ""),
                       "fields": [{"key": f.key, "label": f.label, "raw": f.raw, "value": (f.raw if f.key == "aadhaar_no" else f.value),
                                   "confidence": f.conf, "bbox": list(f.bbox),
                                   "matched_by": f.match, "label_score": f.label_sim} for f in d.fields]} for d in docs],
        "findings": [f.to_dict() for f in findings],
        "capabilities": [{"capability": a, "status": b} for a, b in CAPABILITIES],
        "audit_trail": audit_events,
    }


def build_pdf(dossier, annotated, path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
    ss = getSampleStyleSheet()
    sm = ss["BodyText"].clone("sm", fontSize=8.5, leading=11)
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm)
    S = []
    from xml.sax.saxutils import escape as E          # all dynamic text is escaped: OCR text may contain < & >
    P = lambda t, st=None: Paragraph(t, st or ss["BodyText"])

    from xml.sax.saxutils import escape
    cell = ss["BodyText"].clone("cell", fontSize=8, leading=10)

    def tbl(rows, widths, head=True):
        rows = [[Paragraph(escape(str(c)), cell) for c in r] for r in rows]       # wrap long cells
        t = Table(rows, colWidths=widths, repeatRows=1 if head else 0)
        st = [("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("FONTSIZE", (0, 0), (-1, -1), 8)]
        if head:
            st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf5"))]
        t.setStyle(TableStyle(st))
        return t

    S += [P(f"Document Investigation Dossier - {E(str(dossier['case_id']))}", ss["Title"]),
          P("Generated by DocForensics. Visual findings are indicators of possible alteration, not proof of fraud.", sm), Spacer(1, 6)]
    S += [P("1. Executive summary", ss["Heading2"]), P(E(dossier["summary"]).replace("\n", "<br/>")),
          P(f"<i>Summary source: {E(str(dossier['summary_source']))}</i>", sm)]
    sev = {"high": 0, "medium": 0, "low": 0}
    for f in dossier["findings"]:
        sev[f["severity"]] += 1
    S += [Spacer(1, 4), tbl([["Findings", "High", "Medium", "Low"], [len(dossier["findings"]), sev["high"], sev["medium"], sev["low"]]],
                            [40 * mm] * 4)]
    S += [P("2. Document overview", ss["Heading2"]),
          tbl([["Document", "Type", "Class. conf.", "Detected category (semantic)", "Fields", "SHA-256 (first 16)"]] +
              [[d["doc_id"], d["type"], d["type_conf"], _cat_cell(d), len(d["fields"]), d["sha256"][:16]] for d in dossier["documents"]],
              [34 * mm, 22 * mm, 18 * mm, 48 * mm, 12 * mm, 38 * mm])]
    S += [P("3. Findings (most severe first)", ss["Heading2"])]
    order = {"high": 0, "medium": 1, "low": 2}
    for f in sorted(dossier["findings"], key=lambda x: (order[x["severity"]], -x["confidence"])):
        refs = "; ".join((f"{r['doc_id']}.{r['field']}={r['value']!r}" if r.get("field") else f"{r['doc_id']} (whole page)")
                         for r in f["refs"])
        S += [P(f"<b>{E(f['fid'])}</b> [{E(f['severity'].upper())} | conf {f['confidence']:.2f} | {E(f['status'].replace('_', ' '))}] {E(f['title'])}", ss["BodyText"]),
              P(E(f["explanation"]), sm), P(f"<i>Evidence:</i> {E(refs)}  <i>Method:</i> {E(f['method'])}", sm), Spacer(1, 4)]
    if not dossier["findings"]:
        S += [P("No findings.")]
    S += [PageBreak(), P("4. Extracted information and evidence maps", ss["Heading2"])]
    for d in dossier["documents"]:
        S += [P(f"<b>{E(d['doc_id'])}</b> ({E(d['type'])})", ss["BodyText"])]
        if d["fields"]:
            S += [tbl([["Field", "Value", "OCR conf."]] + [[f["key"], str(f["raw"]), f"{f['confidence']:.2f}"] for f in d["fields"]],
                      [40 * mm, 100 * mm, 25 * mm])]
        else:
            S += [P(("No fields extracted: page not recognised as a supported document type. " if d["type"] == "unknown"
                     else "No label:value fields on this page. ") + E(d.get("decision") or ""), sm)]
        img = annotated.get(d["doc_id"])
        if img:
            S += [Spacer(1, 4), RLImage(img, width=85 * mm, height=85 * mm * 1300 / 1000 * d["size"][1] / 1300 * 1000 / d["size"][0])]
        S += [Spacer(1, 8)]
    S += [PageBreak(), P("5. Capabilities, limitations and areas requiring additional verification", ss["Heading2"]),
          tbl([["Capability", "Status"]] + [[c["capability"], c["status"]] for c in dossier["capabilities"]], [100 * mm, 70 * mm]),
          Spacer(1, 6),
          P("Limitations: peer-relative statistics need at least four extracted fields per page; heavily re-compressed or "
            "re-scanned documents reduce visual-forensic sensitivity; OCR errors can create false content mismatches. "
            "All flagged items should be verified against the original documents and the issuing authority.", sm),
          P("6. Audit trail (summary)", ss["Heading2"]),
          P(f"Hash-chained event log. Digest of the final event (SHA-256): {E(str(dossier.get('audit_digest', ''))[:64])}", sm),
          tbl([["Time", "Event", "Detail"]] + [[e["ts"][11:19], e["event"], str({k: v for k, v in e.items() if k not in ("ts", "event", "prev", "hash")})[:110]]
                                              for e in dossier["audit_trail"][:60]], [20 * mm, 24 * mm, 126 * mm])]
    doc.build(S)
    return path
