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
    ("Section detection; transaction and earnings/deductions tables with running-balance, page-continuity, total and component-sum checks", "Implemented (bank-statement and salary-slip layouts)"),
    ("Evidence mapping (annotated regions), audit trail, reviewer UI, PDF/JSON dossier", "Implemented"),
    ("Handwriting recognition", "Unsupported"),
    ("Typography / spacing / alignment checks, JPEG block-grid consistency (inserted or replaced content), photo-region check on ID cards, signature detection with placement and background checks", "Implemented (heuristic; see limitations)"),
    ("Born-digital PDFs: exact text layer (no OCR), font consistency of amounts, bordered patches over values, covered or hidden text, file history (revisions, dates, producer, Info vs XMP records, font subset tags)", "Implemented (observations about file structure, not proof)"),
    ("Duplicated (copy-move) image regions on scans and photos; image metadata (editing software, modification after capture); scanned-PDF file history", "Implemented (observations; copies recompressed below about JPEG quality 92 after pasting can be missed)"),
    ("Signature authenticity, photo identity matching", "Planned"),
    ("Layout-aware ML models (LayoutLM etc.), tables in other layouts, multilingual OCR", "Planned"),
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


def _structure_story(dossier, structure, P, sm, tbl, mm, RLImage, Spacer, PageBreak, E, ss):
    """Section 3b: detected sections and tables (rows capped; the JSON export has everything)."""
    docs = [d for d in dossier["documents"] if d.get("structure") and (d["structure"]["sections"] or d["structure"]["tables"])]
    if not docs:
        return []
    S = [PageBreak(), P("Document structure: sections and tables", ss["Heading2"]),
         P("Sections are found from headings; tables from their header row. Values are read from the OCR word boxes; checks only use rows "
           "whose amounts could be read.", sm)]
    for d in docs:
        st = d["structure"]
        S += [Spacer(1, 6), P(f"<b>{E(d['doc_id'])}</b> ({E(d['type'])})", ss["BodyText"])]
        lay = st.get("layout")
        if lay:
            h = lay["hierarchy"]
            S += [P(E(f"Layout: {lay['label']}. Hierarchy: page > {h['sections']} section(s) > {h['fields']} field(s), {h['tables']} table(s) "
                      f"({h['table_rows']} rows, {h['table_cells']} cells). Evidence: {'; '.join(lay['evidence'])}."), sm)]
        if st["sections"]:
            S += [tbl([["Section", "Title", "Kind", "Fields inside", "Tables"]] +
                      [[s["sid"], s["title"], s["kind"], ", ".join(s["fields"]) or "-", ", ".join(s["tables"]) or "-"] for s in st["sections"]],
                      [14 * mm, 62 * mm, 32 * mm, 48 * mm, 18 * mm])]
        for t in st["tables"]:
            chk = t["checks"]
            if t["kind"] == "transactions":
                head = (f"Table {t['tid']}: transactions, {len(t['rows'])} rows; running balance checked on {chk['checked']} rows, "
                        f"{len(chk['breaks'])} break(s), {chk['unreadable_rows']} row(s) with unreadable amounts.")
                rows = [["#", "Date", "Description", "Withdrawal", "Deposit", "Balance"]]
                for r in t["rows"][:22]:
                    f = lambda v: "" if v is None else f"{v:,.2f}"
                    rows.append([r["n"], (r["date"] or "") + ("*" if r["inherited_date"] else ""), r["desc"][:44], f(r["debit"]), f(r["credit"]), f(r["balance"])])
                widths = [9 * mm, 24 * mm, 70 * mm, 24 * mm, 24 * mm, 26 * mm]
            else:
                head = f"Table {t['tid']}: earnings / deductions components, {len(t['rows'])} rows; {len(chk['mismatches'])} total mismatch(es)."
                rows = [["#", "Side", "Item", "Amount", "Role"]] + [[r["n"], r["side"], r["label"][:40], f"{r['value']:,.2f}", r["role"]] for r in t["rows"][:22]]
                widths = [9 * mm, 28 * mm, 80 * mm, 30 * mm, 20 * mm]
            S += [Spacer(1, 4), P(E(head), sm), tbl(rows, widths)]
            if len(t["rows"]) > 22:
                S += [P(f"... {len(t['rows']) - 22} more rows in dossier.json. (* = date inherited from the row above)", sm)]
        img = structure.get(d["doc_id"])
        if img and st["tables"]:
            S += [Spacer(1, 4), RLImage(img, width=85 * mm, height=85 * mm * d["size"][1] / d["size"][0])]
    return S


from .report import (CONF_SCALE, confidence_level, group_of, processing_summary, verification_areas, source_info, processing_status, contextual_lists,
                     visual_groups, anomaly_rows, MISSING_M, DUP_M, MISPLACED_M, STANDARDS, STANDARDS_NOTE)


def build_json(case_id, docs, findings, summary, summary_src, audit_events, audit_digest="", correlation=None, evidence=None, audit_info=None):
    from .structure import structure_dict
    return {
        "case_id": case_id, "audit_digest": audit_digest, "summary": summary, "summary_source": summary_src,
        "documents": [{"doc_id": d.doc_id, "source": source_info(d, docs), "processing_status": processing_status(d), "type": d.doc_type, "type_conf": d.type_conf, "sha256": d.sha256,
                       "size": [d.width, d.height], "decision": d.class_info.get("decision", ""),
                       "structure": structure_dict(d), "visual_ext": {"typography": d.facts.get("typography"), "grid": d.facts.get("grid"), "photo": d.facts.get("photo"), "signature": d.facts.get("signature"), "regions": d.facts.get("regions", [])}, "best_guess": d.best_guess, "category_page": d.class_info.get("category"), "category_file": d.class_info.get("file_category"),
                       "skew_deg": d.skew_deg, "born_digital": d.born_digital, "forensics": d.facts.get("forensics", ""),
                       "fields": [{"key": f.key, "label": f.label, "raw": f.raw, "value": (f.raw if f.key == "aadhaar_no" else f.value),
                                   "confidence": f.conf, "bbox": list(f.bbox),
                                   "matched_by": f.match, "label_score": f.label_sim} for f in d.fields]} for d in docs],
        "findings": [{**f.to_dict(), "confidence_level": confidence_level(f.confidence), "group": group_of(f)} for f in findings],
        "confidence_scale": CONF_SCALE,
        "contextual": contextual_lists(findings),
        "visual_groups": visual_groups(docs, findings),
        "anomalies": anomaly_rows(findings),
        "standards": {"note": STANDARDS_NOTE, "rows": [{"framework": a, "what_we_do": b, "what_we_do_not": c} for a, b, c in STANDARDS]},
        "processing": processing_summary(docs, findings, correlation or {}),
        "verification": verification_areas(docs, findings, correlation or {}),
        "correlation": correlation or {},
        "evidence": evidence or {},
        "audit": audit_info or {},
        "capabilities": [{"capability": a, "status": b} for a, b in CAPABILITIES],
        "audit_trail": audit_events,
    }


def build_pdf(dossier, annotated, path, structure=None):
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
          P("Generated by DocForensics. Visual findings are indicators of possible alteration, not proof of fraud. Sections follow the investigation brief (7.1 to 7.6).", sm), Spacer(1, 6)]
    sev = {"high": 0, "medium": 0, "low": 0}
    for f in dossier["findings"]:
        if f.get("group") != "notes":
            sev[f["severity"]] += 1
    S += [P("Case at a glance", ss["Heading2"]),
          tbl([["Documents", "Findings", "High", "Medium", "Low"], [len(dossier["documents"]), sum(sev.values()), sev["high"], sev["medium"], sev["low"]]], [34 * mm] * 5), Spacer(1, 4)]
    S += [P("- " + E(x), sm) for x in dossier.get("processing", [])]
    # ---------------------------------------------------------------- 7.1
    S += [P("7.1 Document Overview", ss["Heading2"]),
          tbl([["Document", "Type", "Source", "Processing status", "SHA-256 (first 16)"]] +
              [[d["doc_id"], f"{d['type']} (class. conf. {d['type_conf']}; {_cat_cell(d)})",
                f"{d['source']['file']} ({d['source']['file_type']}, {d['source']['file_bytes'] or '?'} bytes), page {d['source']['page']} of {d['source']['pages_in_file']}, "
                f"{d['source']['page_px'][0]}x{d['source']['page_px'][1]} px", d["processing_status"], d["sha256"][:16]] for d in dossier["documents"]],
              [24 * mm, 30 * mm, 40 * mm, 56 * mm, 20 * mm])]
    # ---------------------------------------------------------------- 7.2
    S += [PageBreak(), P("7.2 Extracted Information", ss["Heading2"]),
          P("Field values with the confidence they were read with (1.00 for text read exactly from a PDF text layer, OCR word confidence for scans and photos) and their location in the "
            "original page: [x0, y0, x1, y1] in pixels of the page image at 200 dpi (divide by 2.78 for PDF points). The document structure follows.", sm)]
    for d in dossier["documents"]:
        S += [P(f"<b>{E(d['doc_id'])}</b> ({E(d['type'])})", ss["BodyText"])]
        if d["fields"]:
            S += [tbl([["Field", "Value", "Read conf.", "Location (page px)"]] + [[f["key"], str(f["raw"]), f"{f['confidence']:.2f}", str([round(v) for v in f["bbox"]])] for f in d["fields"]],
                      [32 * mm, 74 * mm, 18 * mm, 46 * mm])]
        else:
            S += [P(("No fields extracted: page not recognised as a supported document type. " if d["type"] == "unknown"
                     else "No label:value fields on this page. ") + E(d.get("decision") or ""), sm)]
        S += [Spacer(1, 6)]
    S += _structure_story(dossier, structure or {}, P, sm, tbl, mm, RLImage, Spacer, PageBreak, E, ss)
    order = {"high": 0, "medium": 1, "low": 2}
    ev_all = dossier.get("evidence") or {}
    ev_dir = os.path.dirname(os.path.abspath(path))
    sha = {d_["doc_id"]: d_["sha256"][:8] for d_ in dossier["documents"]}
    LEG = ("<b>How to read the status:</b> <font color='#dc1e1e'>potential alteration</font> = an indicator that something may have been changed, not proof; "
           "<font color='#f08c00'>inconsistency</font> = a confirmed data conflict between values in the documents, which says nothing about intent; "
           "<font color='#286ec8'>observation</font> = a note, coverage gap or weak signal. Nothing here is a finding of fraud.")
    by_fid = {f["fid"]: f for f in dossier["findings"]}

    def finding_blocks(fs):
        out = []
        for f in sorted(fs, key=lambda x: (order[x["severity"]], -x["confidence"])):
            refs = "; ".join((f"{r['doc_id']}[{sha.get(r['doc_id'], '')}].{r['field']}={r['value']!r}" if r.get("field") else f"{r['doc_id']}[{sha.get(r['doc_id'], '')}] (whole page)") for r in f["refs"])
            out += [P(f"<b>{E(f['fid'])}</b> [{E(f['severity'].upper())} | confidence {E(f.get('confidence_level', ''))} ({f['confidence']:.2f}) | {E(f['status'].replace('_', ' '))}] {E(f['title'])}", ss["BodyText"]),
                    P("<b>Explanation:</b> " + E(f["explanation"]), sm), P(f"<i>Evidence:</i> {E(refs)}  <i>Method:</i> {E(f['method'])}", sm)]
            ev = ev_all.get(f["fid"])
            if ev:
                out += [P(f"<b>Status:</b> {E(ev['label'])}", sm), P(f"<b>Why it needs a look:</b> {E(ev['why'])}", sm), P(f"<b>Suggested check:</b> {E(ev['check'])}", sm)]
                cp = os.path.join(ev_dir, ev["crop"]) if ev.get("crop") else None
                if cp and os.path.exists(cp):
                    im_ = Image.open(cp)
                    wmm = min(170.0, im_.width * 0.2646 * 0.75)
                    out += [Spacer(1, 3), RLImage(cp, width=wmm * mm, height=wmm * mm * im_.height / im_.width)]
                elif not ev.get("crop"):
                    out += [P("<i>No region to highlight: this finding concerns the whole page or a value that could not be located.</i>", sm)]
            out += [Spacer(1, 6)]
        return out or [P("None identified.", sm)]
    # ---------------------------------------------------------------- 7.3
    ctx = dossier.get("contextual") or {}
    special = {x["fid"] for k in ("missing", "duplicated", "misplaced") for x in ctx.get(k, [])}
    S += [PageBreak(), P("7.3 Contextual Analysis", ss["Heading2"]), P(LEG, sm), Spacer(1, 4),
          P("Identified field-level inconsistencies", ss["Heading3"]),
          P("Rules that must hold inside one document: arithmetic, totals and running balances, formats, checksums, date logic.", sm)]
    S += finding_blocks([f for f in dossier["findings"] if f.get("group") == "inconsistencies" and f["fid"] not in special])
    S += [P("Missing or duplicated information", ss["Heading3"]),
          P("Expected fields that were absent or unreadable, the same field printed twice with different values, and values that sit under the wrong label.", sm)]
    S += finding_blocks([by_fid[x["fid"]] for k in ("missing", "duplicated", "misplaced") for x in ctx.get(k, []) if x["fid"] in by_fid])
    corr = dossier.get("correlation") or {}
    S += [P("Conflicting information across documents", ss["Heading3"])]
    if corr.get("matrix"):
        S += [P("Every field present in more than one document, with the value in each. 'agree' includes written-differently-but-compatible values (initials, omitted middle name, suffixes).", sm)]
        VERD = {"agree": "agree", "differ": "DIFFER", "single": "one document only", "unverifiable": "could not compare"}
        S += [tbl([["Field", "Result", "Values by document"]] + [[m["field"], VERD.get(m["verdict"], m["verdict"]),
                   "; ".join(f"{e['doc_id']}: {e['raw']}" for e in m["entries"])[:230]] for m in corr["matrix"]], [40 * mm, 28 * mm, 102 * mm])]
        if corr.get("links"):
            S += [Spacer(1, 4), P("Documents linked by shared identifiers: " + E("; ".join(f"{l['a']} - {l['b']} ({', '.join(l['shared'])})" for l in corr["links"])[:900]), sm)]
        if corr.get("unlinked"):
            S += [P("Documents sharing no identifier with the others: " + E(", ".join(corr["unlinked"])), sm)]
    else:
        S += [P("No field appears in more than one document.", sm)]
    S += [P("Cross-document discrepancies", ss["Heading3"])]
    S += finding_blocks([f for f in dossier["findings"] if f.get("group") == "cross"])
    # ---------------------------------------------------------------- 7.4
    S += [PageBreak(), P("7.4 Visual Forensic Findings", ss["Heading2"]),
          P("Pixel-level and PDF-structure indicators that a region may differ from the rest of the page. They show where to look; they do not prove alteration. "
            "Each group says what was checked, so a group without findings states what was examined and found nothing.", sm), Spacer(1, 4)]
    for g in dossier.get("visual_groups", []):
        S += [P(E(g["title"]), ss["Heading3"])]
        if g["checked"]:
            S += [P("<i>Checked:</i> " + E(" | ".join(g["checked"]))[:700], sm)]
        S += finding_blocks([by_fid[x] for x in g["findings"] if x in by_fid])
    # ---------------------------------------------------------------- 7.5
    S += [PageBreak(), P("7.5 Evidence and Traceability", ss["Heading2"]),
          P("Each finding is tied to its original documents (file hash), the extracted fields and values it rests on, and the region it points to (box as in 7.2). Zoomed crops are printed "
            "with each finding in 7.3 and 7.4; the highlighted pages follow the table.", sm)]
    rows = [["ID", "Documents [SHA-256 first 8]", "Associated extracted information", "Region [x0,y0,x1,y1]", "Status", "Confidence"]]
    for f in sorted([f for f in dossier["findings"] if f.get("group") != "notes"], key=lambda x: (order[x["severity"]], -x["confidence"])):
        docs_ = ", ".join(f"{i}[{sha.get(i, '')}]" for i in sorted({r["doc_id"] for r in f["refs"]}))
        info = "; ".join(f"{r['doc_id']}.{r['field']}={r['value']}" for r in f["refs"] if r.get("field"))[:150] or "(whole page)"
        box = "; ".join(str([round(v) for v in r["bbox"]]) for r in f["refs"] if r.get("bbox"))[:120] or "-"
        rows.append([f["fid"], docs_, info, box, f["status"].replace("_", " "), f"{f.get('confidence_level', '')} ({f['confidence']:.2f})"])
    S += [tbl(rows, [12 * mm, 36 * mm, 50 * mm, 36 * mm, 20 * mm, 16 * mm])] if len(rows) > 1 else [P("No findings about the documents.", sm)]
    S += [Spacer(1, 6), P("<b>Highlighted regions</b> (boxes are labelled with finding IDs; colours: red potential alteration, orange inconsistency, blue observation)", ss["BodyText"])]
    imgs = [(d["doc_id"], annotated.get(d["doc_id"]), d) for d in dossier["documents"] if annotated.get(d["doc_id"])]
    cells = []
    for did, img, d in imgs:
        w_ = 80 * mm
        cells.append([RLImage(img, width=w_, height=w_ * d["size"][1] / d["size"][0]), P(E(did), sm)])
    for i in range(0, len(cells), 2):
        pair = cells[i:i + 2]
        S += [Table([[c[0] for c in pair], [c[1] for c in pair]], colWidths=[85 * mm] * len(pair)), Spacer(1, 4)]
    # ---------------------------------------------------------------- 7.6
    S += [PageBreak(), P("7.6 Investigation Summary", ss["Heading2"]), P(E(dossier["summary"]).replace("\n", "<br/>")),
          P(f"<i>Summary source: {E(str(dossier['summary_source']))}</i>", sm), P("Identified anomalies", ss["Heading3"])]
    an = dossier.get("anomalies", [])
    S += [tbl([["ID", "Anomaly", "Severity", "Status", "Confidence", "Documents"]] + [[a_["fid"], a_["title"], a_["severity"], a_["status"].replace("_", " "), a_["confidence"], ", ".join(a_["documents"])] for a_ in an],
              [12 * mm, 62 * mm, 16 * mm, 26 * mm, 20 * mm, 34 * mm])] if an else [P("No anomalies identified. This does not prove that the documents are genuine.", sm)]
    S += [P("Supporting evidence", ss["Heading3"]),
          P("Every anomaly above rests on the references, zoomed crops and highlighted regions in 7.3 to 7.5, and on the hash-chained audit log (appendix C), which records the "
            "references, the SHA-256 of the source file and of the evidence image, and the processing events that produced them.", sm),
          P("Potential explanations", ss["Heading3"]),
          P("Possible, non-exclusive explanations to rule in or out when checking the originals. They are hypotheses, not conclusions.", sm)]
    S += [tbl([["ID", "Possible explanations"]] + [[a_["fid"], " · ".join((ev_all.get(a_["fid"]) or {}).get("explanations", []))] for a_ in an], [14 * mm, 156 * mm])] if an else []
    S += [P("Areas requiring additional verification", ss["Heading3"]),
          tbl([["Area", "Why it needs verification"]] + [[a_["area"], a_["detail"]] for a_ in dossier.get("verification", [])], [60 * mm, 110 * mm]),
          P("Investigation limitations", ss["Heading3"]),
          P("Peer-relative statistics need at least four extracted fields per page. Heavily re-compressed or re-scanned documents reduce visual-forensic sensitivity. OCR errors can create "
            "false content mismatches. Skilled forgeries that match font, ink, tone and keep all numbers consistent are not detected by pixel forensics. Generated PDFs get structure checks only. "
            "No document was verified with its issuer. All flagged items should be verified against the original documents and the issuing authority.", sm),
          P(E(dossier.get("confidence_scale", "")), sm)]
    # ---------------------------------------------------------------- appendices
    au = dossier.get("audit") or {}
    S += [PageBreak(), P("Appendix A. Capabilities and limitations of this system", ss["Heading2"]),
          tbl([["Capability", "Status"]] + [[c["capability"], c["status"]] for c in dossier["capabilities"]], [100 * mm, 70 * mm]),
          Spacer(1, 6),
          P("Appendix B. Standards and reference frameworks (no compliance claim)", ss["Heading2"]),
          P(E(dossier.get("standards", {}).get("note", "")), sm),
          tbl([["Framework", "What we do", "What we do not do"]] + [[r_["framework"], r_["what_we_do"], r_["what_we_do_not"]] for r_ in dossier.get("standards", {}).get("rows", [])], [40 * mm, 65 * mm, 65 * mm]),
          Spacer(1, 6),
          P("Appendix C. Audit trail and investigation history", ss["Heading2"]),
          P(f"Hash-chained event log of run {E(str(au.get('run_id', '')))}: {au.get('events', len(dossier['audit_trail']))} events, chain "
            f"<b>{'verified intact' if au.get('chain_ok') else 'NOT VERIFIED'}</b> (checked in memory and as written to disk). Digest of the final event (SHA-256): "
            f"{E(str(dossier.get('audit_digest', ''))[:64])}", sm),
          P("Each finding record in the log carries its references, the SHA-256 of the source file and of the evidence crop, and the hashes of the ingest, OCR/text-layer, "
            "extraction and field-report events that produced it. References are shown as document[first 8 characters of the file's SHA-256]. "
            "Reviewer decisions are kept in review_log.jsonl (separate hash chain, anchored to this run's digest).", sm),
          P("Earlier runs of this case: " + (E("; ".join(f"{h['run_id']} ({h.get('finished', '')[:16]}, {h.get('findings')} findings, digest {str(h.get('audit_digest', ''))[:12]})" for h in au.get("earlier_runs", []))) or "none (first run)"), sm),
          tbl([["Time", "Event", "Detail"]] + [[e["ts"][11:19], e["event"], str({k: v for k, v in e.items() if k not in ("ts", "event", "prev", "hash")})[:110]]
                                              for e in dossier["audit_trail"][:60]], [20 * mm, 24 * mm, 126 * mm])]
    doc.build(S)
    return path
