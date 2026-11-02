"""Optional LLM reasoning layer. The LLM only *explains* findings that the deterministic
pipeline already produced; it never creates findings, values or evidence.
Falls back to a template summary when no ANTHROPIC_API_KEY / SDK is present."""
import os, json

MODEL = os.environ.get("DOCFOR_MODEL", "claude-sonnet-5-5")
SYSTEM = ("You are a document-fraud analyst assistant. You receive structured findings from an automated "
          "pipeline. Write a concise investigator summary using ONLY the facts provided. Never invent values, "
          "documents or evidence. Separate (a) observed facts, (b) detected inconsistencies, (c) potential "
          "manipulation (indicators, not proof), and list what should be verified next. Max 180 words.")


def _what_it_is(unk):
    """'Semantic analysis: it most resembles a Corporate financial report (...)' when the model was available."""
    from .embed_classifier import describe
    cats = {}
    for d in unk:
        cat = d.class_info.get("file_category") or d.class_info.get("category")
        if cat:
            cats[describe(cat)] = cats.get(describe(cat), 0) + 1
    if not cats:
        return ""
    return "Semantic analysis: it most resembles " + "; ".join(cats) + ", which is outside the supported types. "


def template_summary(case_id, docs, findings):
    n = {"high": 0, "medium": 0, "low": 0}
    for f in findings:
        n[f.severity] += 1
    unk = [d for d in docs if d.doc_type == "unknown"]
    if unk and len(unk) == len(docs):
        return (f"Case {case_id}: none of the {len(docs)} submitted page(s) was recognised as a supported document type "
                f"(application form, ID card, salary slip, bank statement), so no forensic or cross-document analysis was run. "
                f"{_what_it_is(unk)}Reason for {unk[0].doc_id}: {unk[0].class_info.get('decision', 'unrecognised')}.")
    rec = [d for d in docs if d.doc_type != "unknown"]
    ran = [d for d in rec if d.facts.get("forensics") == "run"]
    not_ran = [d for d in rec if d.facts.get("forensics") != "run"]
    cov = (f" Visual forensics ran on {len(ran)} of {len(rec)} recognised page(s)" +
           (f"; it could not run on {len(not_ran)} ({'; '.join(sorted({str(d.facts.get('forensics', '?')) for d in not_ran}))})" if not_ran else "") + ".")
    if not findings or all(f.kind == "missing" and f.fid.startswith("N-") for f in findings):
        if rec and not ran:                       # pixel forensics never ran: do not claim anything about visual anomalies
            return (f"Case {case_id}: {len(docs)} documents analysed; no content inconsistencies were detected by the content and "
                    f"cross-document checks." + cov + " No visual-tampering check was possible, so this is not evidence of authenticity; "
                    "verify against the originals.")
        return (f"Case {case_id}: {len(docs)} documents analysed; no content inconsistencies or visual anomalies were detected." + cov +
                " Absence of findings does not prove authenticity.")
    top = sorted(findings, key=lambda f: ({"high": 0, "medium": 1, "low": 2}[f.severity], -f.confidence))[:4]
    lines = "; ".join(f"{f.fid} {f.title}" for f in top)
    return (f"Case {case_id}: {len(docs)} documents analysed; {len(findings)} findings "
            f"({n['high']} high, {n['medium']} medium, {n['low']} low). Most significant: {lines}. "
            f"Visual findings are indicators of possible alteration, not proof; confirm against originals." + cov)


def llm_summary(case_id, docs, findings, use_llm=False):
    """The LLM summary is OPT-IN per run: it sends finding text (names, numbers, dates from the documents) to an external API."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not use_llm or not key:
        return template_summary(case_id, docs, findings), "template"
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        payload = {"case": case_id, "documents": [{"id": d.doc_id, "type": d.doc_type} for d in docs],
                   "findings": [{k: v for k, v in f.to_dict().items() if k != "metrics"} for f in findings]}
        msg = client.messages.create(model=MODEL, max_tokens=500, system=SYSTEM,
                                     messages=[{"role": "user", "content": json.dumps(payload, default=str)}])
        return msg.content[0].text.strip(), f"llm:{MODEL}"
    except Exception as e:                      # never fail the investigation because of the LLM
        return template_summary(case_id, docs, findings) + f" (LLM unavailable: {type(e).__name__})", "template"
