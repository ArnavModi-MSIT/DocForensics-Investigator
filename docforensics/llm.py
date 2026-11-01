"""Optional LLM reasoning layer. The LLM only *explains* findings that the deterministic
pipeline already produced; it never creates findings, values or evidence.
Falls back to a template summary when no ANTHROPIC_API_KEY / SDK is present."""
import os, json

MODEL = os.environ.get("DOCFOR_MODEL", "claude-sonnet-5-5")
SYSTEM = ("You are a document-fraud analyst assistant. You receive structured findings from an automated "
          "pipeline. Write a concise investigator summary using ONLY the facts provided. Never invent values, "
          "documents or evidence. Separate (a) observed facts, (b) detected inconsistencies, (c) potential "
          "manipulation (indicators, not proof), and list what should be verified next. Max 180 words.")


def template_summary(case_id, docs, findings):
    n = {"high": 0, "medium": 0, "low": 0}
    for f in findings:
        n[f.severity] += 1
    if not findings:
        return (f"Case {case_id}: {len(docs)} documents analysed; no inconsistencies or visual anomalies were "
                f"detected by the automated checks. This does not prove authenticity.")
    top = sorted(findings, key=lambda f: ({"high": 0, "medium": 1, "low": 2}[f.severity], -f.confidence))[:4]
    lines = "; ".join(f"{f.fid} {f.title}" for f in top)
    return (f"Case {case_id}: {len(docs)} documents analysed; {len(findings)} findings "
            f"({n['high']} high, {n['medium']} medium, {n['low']} low). Most significant: {lines}. "
            f"Visual findings are indicators of possible alteration, not proof; confirm against originals.")


def llm_summary(case_id, docs, findings):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
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
