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
    ("Born-digital PDFs: exact text layer (no OCR), font consistency of amounts, bordered patches over values, covered or hidden text, file history (revisions, dates, producer)", "Implemented (observations about file structure, not proof)"),
    ("Signature authenticity, photo identity matching, copy-move detection", "Planned"),
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
    S = [PageBreak(), P("3b. Document structure: sections and tables", ss["Heading2"]),
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


from .report import CONF_SCALE, confidence_level, group_of, processing_summary, verification_areas


def build_json(case_id, docs, findings, summary, summary_src, audit_events, audit_digest="", correlation=None, evidence=None, audit_info=None):
    from .structure import structure_dict
    return {
        "case_id": case_id, "audit_digest": audit_digest, "summary": summary, "summary_source": summary_src,
        "documents": [{"doc_id": d.doc_id, "type": d.doc_type, "type_conf": d.type_conf, "sha256": d.sha256,
                       "size": [d.width, d.height], "decision": d.class_info.get("decision", ""),
                       "structure": structure_dict(d), "visual_ext": {"typography": d.facts.get("typography"), "grid": d.facts.get("grid"), "photo": d.facts.get("photo"), "signature": d.facts.get("signature"), "regions": d.facts.get("regions", [])}, "best_guess": d.best_guess, "category_page": d.class_info.get("category"), "category_file": d.class_info.get("file_category"),
                       "skew_deg": d.skew_deg, "born_digital": d.born_digital, "forensics": d.facts.get("forensics", ""),
                       "fields": [{"key": f.key, "label": f.label, "raw": f.raw, "value": (f.raw if f.key == "aadhaar_no" else f.value),
                                   "confidence": f.conf, "bbox": list(f.bbox),
                                   "matched_by": f.match, "label_score": f.label_sim} for f in d.fields]} for d in docs],
        "findings": [{**f.to_dict(), "confidence_level": confidence_level(f.confidence), "group": group_of(f)} for f in findings],
        "confidence_scale": CONF_SCALE,
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
          P("Generated by DocForensics. Visual findings are indicators of possible alteration, not proof of fraud.", sm), Spacer(1, 6)]
    S += [P("1. Executive summary and what was checked", ss["Heading2"]), P(E(dossier["summary"]).replace("\n", "<br/>")),
          P(f"<i>Summary source: {E(str(dossier['summary_source']))}</i>", sm)]
    sev = {"high": 0, "medium": 0, "low": 0}
    for f in dossier["findings"]:
        sev[f["severity"]] += 1
    S += [Spacer(1, 4), tbl([["Findings", "High", "Medium", "Low"], [len(dossier["findings"]), sev["high"], sev["medium"], sev["low"]]], [40 * mm] * 4), Spacer(1, 4)]
    S += [P("<b>What was checked in this case</b>", ss["BodyText"])] + [P("- " + E(x), sm) for x in dossier.get("processing", [])]
    S += [P("2. Document overview", ss["Heading2"]),
          tbl([["Document", "Type", "Class. conf.", "Detected category (semantic)", "Fields", "SHA-256 (first 16)"]] +
              [[d["doc_id"], d["type"], d["type_conf"], _cat_cell(d), len(d["fields"]), d["sha256"][:16]] for d in dossier["documents"]],
              [34 * mm, 22 * mm, 18 * mm, 48 * mm, 12 * mm, 38 * mm])]
    S += [PageBreak(), P("3. Extracted information", ss["Heading2"]),
          P("Every field with its value and the confidence with which it was read (1.00 for text read exactly from a PDF text layer; OCR word confidence for "
            "scans and photos). Page images below show the regions that findings point to.", sm)]
    for d in dossier["documents"]:
        S += [P(f"<b>{E(d['doc_id'])}</b> ({E(d['type'])})", ss["BodyText"])]
        if d["fields"]:
            S += [tbl([["Field", "Value", "Read confidence"]] + [[f["key"], str(f["raw"]), f"{f['confidence']:.2f}"] for f in d["fields"]],
                      [40 * mm, 100 * mm, 25 * mm])]
        else:
            S += [P(("No fields extracted: page not recognised as a supported document type. " if d["type"] == "unknown"
                     else "No label:value fields on this page. ") + E(d.get("decision") or ""), sm)]
        img = annotated.get(d["doc_id"])
        if img:
            S += [Spacer(1, 4), RLImage(img, width=85 * mm, height=85 * mm * 1300 / 1000 * d["size"][1] / 1300 * 1000 / d["size"][0])]
        S += [Spacer(1, 8)]
    S += _structure_story(dossier, structure or {}, P, sm, tbl, mm, RLImage, Spacer, PageBreak, E, ss)
    order = {"high": 0, "medium": 1, "low": 2}
    ev_all = dossier.get("evidence") or {}
    ev_dir = os.path.dirname(os.path.abspath(path))
    LEG = ("<b>How to read the status:</b> <font color='#dc1e1e'>potential alteration</font> = an indicator that something may have been changed, not proof; "
           "<font color='#f08c00'>inconsistency</font> = a confirmed data conflict between values in the documents, which says nothing about intent; "
           "<font color='#286ec8'>observation</font> = a note, coverage gap or weak signal. Nothing here is a finding of fraud.")

    def finding_blocks(group):
        fs = sorted([f for f in dossier["findings"] if f.get("group") == group], key=lambda x: (order[x["severity"]], -x["confidence"]))
        out = []
        for f in fs:
            sha = {d_["doc_id"]: d_["sha256"][:8] for d_ in dossier["documents"]}
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
    S += [PageBreak(), P("4. Identified inconsistencies within documents", ss["Heading2"]),
          P(LEG, sm), P("Each finding below carries its explanation, supporting evidence (references and a zoomed crop), confidence level and a suggested check.", sm), Spacer(1, 4)]
    S += finding_blocks("inconsistencies")
    S += [P("5. Visual forensic observations", ss["Heading2"]),
          P("Pixel-level and PDF-structure indicators that a region may differ from the rest of the page. They show where to look; they do not prove alteration.", sm), Spacer(1, 4)]
    S += finding_blocks("visual")
    S += [P("6. Cross-document discrepancies", ss["Heading2"])]
    corr = dossier.get("correlation") or {}
    if corr.get("matrix"):
        S += [P("Every field present in more than one document, with the value in each and the result. 'agree' includes written-differently-but-compatible "
                "values (initials, omitted middle name, suffixes). Source boxes are in dossier.json.", sm)]
        VERD = {"agree": "agree", "differ": "DIFFER", "single": "one document only", "unverifiable": "could not compare"}
        S += [tbl([["Field", "Result", "Values by document"]] + [[m["field"], VERD.get(m["verdict"], m["verdict"]),
                   "; ".join(f"{e['doc_id']}: {e['raw']}" for e in m["entries"])[:230]] for m in corr["matrix"]], [40 * mm, 28 * mm, 102 * mm])]
        if corr.get("links"):
            S += [Spacer(1, 4), P("Documents linked by shared identifiers: " + E("; ".join(f"{l['a']} - {l['b']} ({', '.join(l['shared'])})" for l in corr["links"])[:900]), sm)]
        if corr.get("unlinked"):
            S += [P("Documents sharing no identifier with the others: " + E(", ".join(corr["unlinked"])), sm)]
    S += [Spacer(1, 6)] + finding_blocks("cross")
    S += [PageBreak(), P("7. Confidence levels", ss["Heading2"]), P(E(dossier.get("confidence_scale", "")), sm), Spacer(1, 4),
          tbl([["ID", "Section", "Severity", "Confidence", "Status"]] +
              [[f["fid"], {"inconsistencies": "within document", "visual": "visual", "cross": "cross-document", "notes": "coverage note"}.get(f.get("group"), ""),
                f["severity"], f"{f.get('confidence_level', '')} ({f['confidence']:.2f})", f["status"].replace("_", " ")]
               for f in sorted(dossier["findings"], key=lambda x: (order[x["severity"]], -x["confidence"]))], [20 * mm, 36 * mm, 24 * mm, 36 * mm, 50 * mm])]
    S += [P("8. Areas requiring additional verification", ss["Heading2"]),
          P("Specific to this case: what could not be read, checked or compared, and what no automated check can establish.", sm)]
    S += [tbl([["Area", "Why it needs verification"]] + [[a_["area"], a_["detail"]] for a_ in dossier.get("verification", [])], [60 * mm, 110 * mm])]
    S += [Spacer(1, 6), P("Limitations of the method: peer-relative statistics need at least four extracted fields per page; heavily re-compressed or re-scanned documents "
                          "reduce visual-forensic sensitivity; OCR errors can create false content mismatches. All flagged items should be verified against the original "
                          "documents and the issuing authority.", sm)]
    au = dossier.get("audit") or {}
    S += [PageBreak(), P("9. Capabilities and limitations of this system", ss["Heading2"]),
          tbl([["Capability", "Status"]] + [[c["capability"], c["status"]] for c in dossier["capabilities"]], [100 * mm, 70 * mm]),
          Spacer(1, 6),
          P("10. Audit trail and investigation history", ss["Heading2"]),
          P(f"Hash-chained event log of run {E(str(au.get('run_id', '')))}: {au.get('events', len(dossier['audit_trail']))} events, chain "
            f"<b>{'verified intact' if au.get('chain_ok') else 'NOT VERIFIED'}</b> (checked in memory and as written to disk). Digest of the final event (SHA-256): "
            f"{E(str(dossier.get('audit_digest', ''))[:64])}", sm),
          P("Each finding record in the log carries its references, the SHA-256 of the source file and of the evidence crop, and the hashes of the ingest, OCR/text-layer, "
            "extraction and field-report events that produced it. In section 4-6 references are shown as document[first 8 characters of the file's SHA-256].field. "
            "Reviewer decisions are kept in review_log.jsonl (separate hash chain, anchored to this run's digest).", sm),
          P("Earlier runs of this case: " + (E("; ".join(f"{h['run_id']} ({h.get('finished', '')[:16]}, {h.get('findings')} findings, digest {str(h.get('audit_digest', ''))[:12]})" for h in au.get("earlier_runs", []))) or "none (first run)"), sm),
          tbl([["Time", "Event", "Detail"]] + [[e["ts"][11:19], e["event"], str({k: v for k, v in e.items() if k not in ("ts", "event", "prev", "hash")})[:110]]
                                              for e in dossier["audit_trail"][:60]], [20 * mm, 24 * mm, 126 * mm])]
    doc.build(S)
    return path
