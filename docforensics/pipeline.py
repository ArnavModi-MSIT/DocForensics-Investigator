import argparse, glob, json, os, shutil
from .audit import AuditTrail
from .schema import Finding
from .ingest import ingest, expand_inputs
from .ocr import run_ocr
from .extract import classify, extract_fields, inherit_continuations, assign_file_categories
from .forensics import analyze, save_ela_heatmap
from .correlate import check_intra, check_cross, corroborate
from .llm import llm_summary
from . import dossier as dz

EXTS = (".jpg", ".jpeg", ".png", ".pdf", ".tif", ".tiff", ".bmp", ".webp", ".zip")


def run_case(paths, outdir, case_id=None, progress=None, use_llm=False):
    """Full investigation. progress(msg) is called per stage (used by the UI)."""
    say = progress or (lambda m: None)
    case_id = case_id or os.path.basename(os.path.normpath(outdir))
    os.makedirs(outdir, exist_ok=True)
    audit_path = os.path.join(outdir, "audit.jsonl")
    open(audit_path, "w").close()          # truncate (do not delete) any previous audit log
    audit = AuditTrail(audit_path)
    audit.log("case_start", case_id=case_id, n_inputs=len(paths))
    work = os.path.join(outdir, "pages")
    say("Ingesting documents")
    paths, skipped = expand_inputs(paths, work, audit)
    docs = ingest(paths, work, audit)
    if not docs:
        raise ValueError("No supported documents found in the input (accepted: images, PDF, ZIP of those).")
    visual, table = [], []
    for d in docs:
        say(f"OCR + extraction: {d.doc_id}")
        run_ocr(d, audit); extract_fields(d, audit); classify(d, audit)
    inherit_continuations(docs, audit)
    assign_file_categories(docs, audit)
    for d in docs:
        if d.doc_type == "unknown":
            continue                          # forensics only on recognised document types
        if d.born_digital:
            d.facts["forensics"] = "not applicable: born-digital PDF page"
            audit.log("forensics_skipped", doc_id=d.doc_id, reason="born-digital PDF page (no scan/capture)")
            continue
        say(f"Visual forensics: {d.doc_id}")
        fs, tb = analyze(d, audit, id_start=len(visual) + 1)
        visual += fs; table += tb
        d.facts["forensics"] = "run" if tb else "not run: " + d.facts.pop("forensics_reason", "too few comparable fields (need >= 4 of similar text size)")
        save_ela_heatmap(d, os.path.join(outdir, f"{d.doc_id}_ela.png"))
    for d in docs:                            # ELA map for every page so the UI can always show it
        pth = os.path.join(outdir, f"{d.doc_id}_ela.png")
        if not os.path.exists(pth):
            save_ela_heatmap(d, pth)
    say("Contextual verification and cross-document correlation")
    others = check_intra(docs) + check_cross(docs)
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
                                "does not apply. Content checks (arithmetic, formats, cross-document) were still run. PDF structure and "
                                "metadata checks are planned. Pages: " + ", ".join(d.doc_id for d in bd[:8]) + (" ..." if len(bd) > 8 else ""),
                                [{"doc_id": d.doc_id, "field": None, "value": None, "bbox": None} for d in bd[:8]],
                                method="input type detection"))
    for f in findings:
        audit.log("finding", fid=f.fid, kind=f.kind, severity=f.severity, confidence=f.confidence, title=f.title)
    say("Generating dossier")
    summary, src = llm_summary(case_id, docs, findings, use_llm=use_llm)
    annotated = {d.doc_id: dz.annotate(d, findings, os.path.join(outdir, f"{d.doc_id}_annotated.png")) for d in docs}
    audit.log("dossier", case_id=case_id, n_findings=len(findings), summary_source=src)
    data = dz.build_json(case_id, docs, findings, summary, src, audit.events, audit.digest)
    jpath = os.path.join(outdir, "dossier.json")
    json.dump(data, open(jpath, "w"), indent=2, default=str)
    try:
        ppath = dz.build_pdf(data, annotated, os.path.join(outdir, "dossier.pdf"))
    except Exception as e:                                   # keep the investigation; the JSON dossier is already saved
        ppath = None
        audit.log("pdf_failed", error=f"{type(e).__name__}: {str(e)[:120]}")
        say(f"PDF dossier could not be built ({type(e).__name__}); JSON dossier saved")
    say("Done")
    return {"docs": docs, "findings": findings, "dossier": data, "json": jpath, "pdf": ppath,
            "annotated": annotated, "forensic_table": table, "outdir": outdir, "skipped": skipped}


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
