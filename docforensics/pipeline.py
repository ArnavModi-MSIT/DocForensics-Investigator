import argparse, glob, json, os, shutil, time
from .audit import AuditTrail, new_run_id, archive_previous, verify_chain, verify_file, load_history, append_history, sha256_file
from .schema import Finding
from .ingest import ingest, expand_inputs
from .ocr import run_ocr
from .extract import classify, extract_fields, inherit_continuations, assign_file_categories, post_classify, issuer_as_employer
from .structure import detect_structure, derive_table_fields, table_findings, save_structure_overlay
from .forensics import analyze, save_ela_heatmap
from .forensics_ext import analyze_ext, analyze_signature_only, save_regions_overlay
from .pdf_forensics import analyze_pdf_docs
from .pdf_layer import restore_pdf_geometry
from .linkage import build_correlation, linkage_findings
from .evidence import build_evidence
from .correlate import check_intra, check_cross, corroborate
from .consistency import consistency_findings
from .llm import llm_summary
from . import dossier as dz

EXTS = (".jpg", ".jpeg", ".png", ".pdf", ".tif", ".tiff", ".bmp", ".webp", ".zip")


def run_case(paths, outdir, case_id=None, progress=None, use_llm=False):
    """Full investigation. progress(msg) is called per stage (used by the UI)."""
    say = progress or (lambda m: None)
    case_id = case_id or os.path.basename(os.path.normpath(outdir))
    os.makedirs(outdir, exist_ok=True)
    audit_path = os.path.join(outdir, "audit.jsonl")
    run_id = new_run_id()
    archive_previous(audit_path, outdir)    # the log of an earlier run of this case is kept under history/ before the file is reused
    earlier_runs = load_history(outdir)
    open(audit_path, "w").close()          # truncate (do not delete) the live log; the archive and case_history.json keep the past
    audit = AuditTrail(audit_path, run_id)
    audit.log("case_start", case_id=case_id, n_inputs=len(paths), run_id=run_id, earlier_runs=len(earlier_runs))
    work = os.path.join(outdir, "pages")
    say("Ingesting documents")
    paths, skipped = expand_inputs(paths, work, audit)
    docs = ingest(paths, work, audit)
    if not docs:
        raise ValueError("No supported documents found in the input (accepted: images, PDF, ZIP of those).")
    visual, table = [], []
    for d in docs:
        say(f"OCR + extraction: {d.doc_id}")
        run_ocr(d, audit); extract_fields(d, audit); classify(d, audit); post_classify(d, audit)
    inherit_continuations(docs, audit)
    assign_file_categories(docs, audit)
    for d in docs:
        if d.doc_type != "unknown":
            say(f"Sections and tables: {d.doc_id}")
            detect_structure(d, audit)
            derive_table_fields(d, audit)
    for d in docs:
        if d.doc_type == "unknown":
            continue                          # forensics only on recognised document types
        if d.born_digital:
            d.facts["forensics"] = "pixel-level checks not applicable (born-digital PDF page); PDF structure checks run instead"
            audit.log("forensics_skipped", doc_id=d.doc_id, reason="born-digital PDF page (no scan/capture); PDF structure checks run instead")
            continue
        say(f"Visual forensics: {d.doc_id}")
        fs, tb = analyze(d, audit, id_start=len(visual) + 1)
        visual += fs; table += tb
        d.facts["forensics"] = "run" if tb else "not run: " + d.facts.pop("forensics_reason", "too few comparable fields (need >= 4 of similar text size)")
        save_ela_heatmap(d, os.path.join(outdir, f"{d.doc_id}_ela.png"))
    for d in docs:                            # typography / alignment, JPEG block grid, photo and signature regions (recognised pages only)
        if d.born_digital:
            continue                          # generated PDF: the text layer and drawing records replace pixel analysis (below)
        if d.doc_type != "unknown":
            say(f"Typography, image regions, signatures: {d.doc_id}")
            visual += analyze_ext(d, audit, id_start=len(visual) + 1)
        else:
            visual += analyze_signature_only(d, audit, id_start=len(visual) + 1)           # unsupported page: signature marks only
    say("PDF structure checks")
    visual += analyze_pdf_docs([d for d in docs if d.doc_type != "unknown"], audit, id_start=len(visual) + 1)
    for d in docs:                            # ELA map for every page so the UI can always show it
        pth = os.path.join(outdir, f"{d.doc_id}_ela.png")
        if not os.path.exists(pth):
            save_ela_heatmap(d, pth)
    issuer_as_employer(docs, audit)
    say("Contextual verification and cross-document correlation")
    others = check_intra(docs) + check_cross(docs) + table_findings(docs) + consistency_findings(docs)
    corroborate(visual, others)
    findings = visual + others
    few = [d for d in docs if d.doc_type != "unknown" and str(d.facts.get("forensics", "")).startswith("not run")]
    if few:
        findings.append(Finding("N-002", "missing", f"Visual forensics could not be run on {len(few)} page(s)", "low", 0.9, "observed",
                                "Pixel-level checks compare a field with other fields of similar text size on the same page and need at least four. "
                                "The absence of visual findings on these pages is NOT evidence of authenticity. Content checks were still run. "
                                "Pages: " + "; ".join(f"{d.doc_id} ({d.facts.get('forensics', '').replace('not run: ', '')})" for d in few[:6]) + (" ..." if len(few) > 6 else ""),
                                [{"doc_id": d.doc_id, "field": None, "value": None, "bbox": None} for d in few[:8]],
                                method="forensics coverage"))
    weak = [d for d in docs if (d.doc_type != "unknown" or d.fields) and (d.facts.get("ocr_mean_conf") or 1) < 0.6]
    if weak:
        from .ocr import rapidocr_available
        findings.append(Finding("N-003", "missing", f"Low OCR confidence on {len(weak)} page(s)", "low", 0.9, "observed",
                                "These pages were read with mean word confidence below 0.60 even after retries, so extracted values may be misread "
                                "(a misread value can cause a false mismatch or hide a real one). Verify the fields manually. "
                                + ("" if rapidocr_available() else "RapidOCR is not installed; installing it usually improves busy or photographed cards. ")
                                + "Pages: " + "; ".join(f"{d.doc_id} ({d.facts.get('ocr_engine')}, {d.facts.get('ocr_mean_conf')})" for d in weak[:6]),
                                [{"doc_id": d.doc_id, "field": None, "value": None, "bbox": None} for d in weak[:8]],
                                method="OCR quality"))
    bd = [d for d in docs if d.born_digital and d.doc_type != "unknown"]
    if bd:
        findings.append(Finding("N-001", "missing", f"Pixel-level forensics not applicable to {len(bd)} born-digital PDF page(s)", "low", 0.9,
                                "observed", "These pages are generated PDFs (text layer, no scanned image), so compression/grain/ink analysis "
                                "does not apply. Their exact text was read from the PDF (no OCR), and PDF structure checks were run instead: font "
                                "consistency of amounts, bordered patches and covered or hidden text, and file history. Content checks (arithmetic, "
                                "formats, cross-document) were run as for any page. A clean result is not proof of authenticity: a document can be "
                                "forged before it is turned into a PDF. Pages: " + ", ".join(d.doc_id for d in bd[:8]) + (" ..." if len(bd) > 8 else ""),
                                [{"doc_id": d.doc_id, "field": None, "value": None, "bbox": None} for d in bd[:8]],
                                method="input type detection"))
    restore_pdf_geometry(docs, findings)        # born-digital pages: report boxes in the PDF's own word geometry
    corr = build_correlation(docs)              # which documents belong together and what was compared (after the geometry restore)
    findings += linkage_findings(docs, corr)
    for d in docs:                              # the fields as reported (final boxes), so every value can be traced to its place on the page
        audit.log("fields_final", doc_id=d.doc_id, doc_type=d.doc_type, source_sha256=d.sha256,
                  fields=[{"key": f.key, "raw": f.raw, "bbox": [round(v, 1) for v in f.bbox], "conf": f.conf, "matched_by": f.match} for f in d.fields])
    evidence = build_evidence(docs, findings, outdir)          # zoomed crops + why / what to check for every finding
    by_sha = {d.doc_id: d.sha256 for d in docs}
    for f in findings:                          # a finding record names its evidence, the source files and the events that produced them
        ev = evidence.get(f.fid, {})
        crop = os.path.join(outdir, ev["crop"]) if ev.get("crop") else None
        ids = sorted({r["doc_id"] for r in f.refs})
        audit.log("finding", fid=f.fid, kind=f.kind, severity=f.severity, confidence=f.confidence, status=f.status, title=f.title, method=f.method,
                  refs=[{"doc_id": r["doc_id"], "field": r.get("field"), "value": r.get("value"), "bbox": r.get("bbox")} for r in f.refs],
                  source_sha256={i: by_sha.get(i) for i in ids}, evidence_crop=ev.get("crop"),
                  evidence_sha256=sha256_file(crop) if crop and os.path.exists(crop) else None,
                  produced_by={i: {e_: audit.find_hash(e_, i) for e_ in ("ingest", "ocr", "extract", "fields_final")} for i in ids})
    say("Generating dossier")
    summary, src = llm_summary(case_id, docs, findings, use_llm=use_llm)
    annotated = {d.doc_id: dz.annotate(d, findings, os.path.join(outdir, f"{d.doc_id}_annotated.png")) for d in docs}
    regions = {d.doc_id: save_regions_overlay(d, os.path.join(outdir, f"{d.doc_id}_regions.png")) for d in docs if d.facts.get("regions")}
    structure = {d.doc_id: save_structure_overlay(d, os.path.join(outdir, f"{d.doc_id}_structure.png")) for d in docs if d.sections or d.tables}
    audit.log("dossier", case_id=case_id, n_findings=len(findings), summary_source=src)
    chain_ok = verify_chain(audit.events) and verify_file(audit_path)[0]          # in memory AND as written to disk
    audit_info = {"run_id": run_id, "chain_ok": chain_ok, "events": len(audit.events), "digest": audit.digest, "earlier_runs": earlier_runs,
                  "review_log": "review_log.jsonl (reviewer decisions, hash-chained, anchored to this digest)"}
    data = dz.build_json(case_id, docs, findings, summary, src, audit.events, audit.digest, correlation=corr, evidence=evidence, audit_info=audit_info)
    jpath = os.path.join(outdir, "dossier.json")
    json.dump(data, open(jpath, "w"), indent=2, default=str)
    try:
        ppath = dz.build_pdf(data, annotated, os.path.join(outdir, "dossier.pdf"), structure=structure)
    except Exception as e:                                   # keep the investigation; the JSON dossier is already saved
        ppath = None
        audit.log("pdf_failed", error=f"{type(e).__name__}: {str(e)[:120]}")
        say(f"PDF dossier could not be built ({type(e).__name__}); JSON dossier saved")
    append_history(outdir, {"run_id": run_id, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "documents": {d.doc_id: d.sha256 for d in docs},
                            "findings": len(findings), "high": sum(f.severity == "high" for f in findings), "audit_events": len(audit.events),
                            "audit_digest": audit.last, "chain_ok": chain_ok, "audit_log": f"history/{run_id}.jsonl (archived when the case is re-run)"})
    say("Done")
    return {"docs": docs, "findings": findings, "dossier": data, "json": jpath, "pdf": ppath,
            "annotated": annotated, "structure": structure, "regions": regions, "forensic_table": table, "correlation": corr, "evidence": evidence, "audit_info": audit_info, "run_id": run_id, "outdir": outdir, "skipped": skipped}


def main():
    ap = argparse.ArgumentParser(description="Run a document forensics investigation on a folder or files.")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--llm", action="store_true", help="send finding text to the Anthropic API for the summary (needs ANTHROPIC_API_KEY)")
    a = ap.parse_args()
    paths = []
    for i in a.inputs:
        paths += sorted(p for p in glob.glob(os.path.join(i, "*")) if p.lower().endswith(EXTS)) if os.path.isdir(i) else [i]
    out = a.out or os.path.join("output", os.path.basename(os.path.normpath(a.inputs[0])))
    r = run_case(paths, out, progress=print, use_llm=a.llm)
    for f in sorted(r["findings"], key=lambda f: f.fid):
        print(f"  {f.fid} [{f.severity:6}|{f.confidence:.2f}|{f.status}] {f.title}")
    print("PDF:", r["pdf"])


if __name__ == "__main__":
    main()
