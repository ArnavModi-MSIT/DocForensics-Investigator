"""DocForensics - Streamlit UI.   Run:  streamlit run app.py"""
import glob, os, re, tempfile, time
import pandas as pd
import streamlit as st
from docforensics.pipeline import run_case
from docforensics.debug import explain
from docforensics.review import record_decision, load_decisions, verify_decisions, DECISIONS, LOG as REVIEW_LOG


def _frame(rows):
    """DataFrame whose mixed-type (text + number) columns are converted to text, so Streamlit's Arrow conversion
    never has to fall back and log a traceback."""
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: "" if x is None else str(x))
    return df


st.set_page_config(page_title="DocForensics", layout="wide")
st.sidebar.text_input("Reviewer name (recorded with your decisions)", key="reviewer")
st.title("DocForensics - Document Forensics & Contextual Intelligence")
st.caption("Visual findings are indicators of possible alteration, not proof. Verify against originals.")

ROOT = os.path.dirname(__file__)
with st.sidebar:
    st.header("Case input")
    mode = st.radio("Source", ["Sample case", "Upload documents"])
    paths = []
    if mode == "Sample case":
        cases = sorted(d for d in glob.glob(os.path.join(ROOT, "samples", "case_*")))
        if cases:
            pick = st.selectbox("Sample", cases, format_func=os.path.basename)
            paths = sorted(glob.glob(os.path.join(pick, "*.jpg")))
            case_id = os.path.basename(pick)
        else:
            st.info("No sample cases yet. They are fictional and generated on demand: run `python synth/generate.py` "
                    "(start.cmd does this on first start), or switch to *Upload documents*.")
            case_id = "case"
    else:
        ups = st.file_uploader("Images / PDFs / ZIP (one zip = one case)", accept_multiple_files=True,
                               type=["jpg", "jpeg", "png", "pdf", "tif", "tiff", "zip"])
        case_id = st.text_input("Case ID", f"case_{int(time.time())}")
        tmp = tempfile.mkdtemp()
        for u in ups or []:
            p = os.path.join(tmp, u.name); open(p, "wb").write(u.getbuffer()); paths.append(p)
    case_id = re.sub(r"[^A-Za-z0-9_.-]", "_", case_id).strip("._")[:64] or "case"      # never use raw user text in a path
    go = st.button("Run investigation", type="primary", disabled=not paths)
    use_llm = st.checkbox("Send findings to Claude for the summary", value=False, disabled=not os.environ.get("ANTHROPIC_API_KEY"),
                          help="Off by default. When on, finding text (names, numbers, dates) is sent to an external API. Needs ANTHROPIC_API_KEY.")

if go:
    bar, msg = st.progress(0.0), st.empty()
    steps = {"n": 0}
    total = 3 * len(paths) + 4

    def prog(m):
        steps["n"] += 1
        bar.progress(min(1.0, steps["n"] / total)); msg.write(m)
    try:
        st.session_state["res"] = run_case(paths, os.path.join(ROOT, "output", case_id), case_id, prog, use_llm=use_llm)
    except ValueError as e:
        st.session_state.pop("res", None); bar.empty(); msg.empty(); st.error(str(e)); st.stop()
    bar.empty(); msg.empty()

res = st.session_state.get("res")
if not res:
    st.info("Pick a sample case or upload related documents, then run the investigation.")
    st.stop()

F, D = res["findings"], res["docs"]
if res.get("skipped"):
    st.warning("Skipped %d file(s) in the upload: " % len(res["skipped"]) + "; ".join(f"{x['name']} ({x['reason']})" for x in res["skipped"][:10]))
from docforensics.report import confidence_level
from docforensics.audit import load_history

dj, ai = res["dossier"], res.get("audit_info") or {}
G = {f["fid"]: f["group"] for f in dj["findings"]}
REAL = [f for f in F if G.get(f.fid) != "notes"]                       # findings about the documents; coverage notes live under "areas requiring verification"
SEV = {"high": 0, "medium": 1, "low": 2}
ICON = {"potential_manipulation": "🟥", "inconsistency": "🟧", "observed": "🟦"}
GUIDE = ("🟥 potential alteration: an indicator, not proof  ·  🟧 inconsistency: a confirmed data conflict between values (says nothing about intent)  ·  "
         "🟦 observation: a note or weak signal.  Nothing here is a finding of fraud.")
_widget_n = [0]


def review_form(f):
    """Reviewer decision for one finding: recorded in a hash-chained log next to the case, anchored to this run's audit digest."""
    _widget_n[0] += 1
    for e in [e for e in load_decisions(res["outdir"], res["run_id"]) if e.get("fid") == f.fid][-2:]:
        st.caption(f"Decision ({e['ts'][:16]}, {e.get('reviewer') or 'anonymous'}): {e['decision']}" + (f" | {e['note']}" if e.get("note") else ""))
    with st.form(f"rv_{f.fid}_{_widget_n[0]}", border=False):
        c1, c2, c3 = st.columns([3, 3, 1])
        dec = c1.selectbox("Your decision", DECISIONS, key=f"d_{f.fid}_{_widget_n[0]}", label_visibility="collapsed")
        note = c2.text_input("Note", key=f"n_{f.fid}_{_widget_n[0]}", placeholder="Note (optional)", label_visibility="collapsed")
        if c3.form_submit_button("Record"):
            record_decision(res["outdir"], res["run_id"], ai["digest"], f.fid, dec, note, st.session_state.get("reviewer", ""))
            st.success("Recorded in the review log.")


def finding_card(f, expanded=False):
    ev = (res.get("evidence") or {}).get(f.fid) or {}
    lvl = confidence_level(f.confidence)
    with st.expander(f"{ICON.get(f.status, '')} **{f.severity.upper()}** · {f.title}", expanded=expanded):
        st.caption(f"{f.fid} · confidence {lvl} ({f.confidence:.2f}) · {f.method}")
        st.write(f.explanation)
        cp = os.path.join(res["outdir"], ev["crop"]) if ev.get("crop") else None
        if cp and os.path.exists(cp):
            left, right = st.columns([3, 2])
            left.image(cp)
            right.markdown(f"**Why it needs a look**  \n{ev['why']}\n\n**Suggested check**  \n{ev['check']}\n\n**Possible explanations**  \n" + "  \n".join("• " + x for x in ev.get("explanations", [])))
        elif ev:
            st.markdown(f"**Why it needs a look:** {ev['why']}  \n**Suggested check:** {ev['check']}")
        if ev:
            st.caption(ev["label"])
        review_form(f)


# ---- header -----------------------------------------------------------------------------------------------------------------
n_high = sum(f.severity == "high" for f in REAL)
n_med = sum(f.severity == "medium" for f in REAL)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents", len(D)); c2.metric("Findings", len(REAL)); c3.metric("High severity", n_high)
c4.metric("Potential alteration", sum(f.status == "potential_manipulation" for f in REAL))
if n_high:
    st.error(f"{n_high} high-severity finding(s) need review before this case is relied on.")
elif n_med:
    st.warning(f"{n_med} medium-severity finding(s) to review.")
else:
    st.success("No inconsistencies or visual anomalies detected. This is not proof that the documents are genuine.")

tab_d, tab_docs, tab_a = st.tabs(["📋 Dossier", "📄 Documents", "🛡 Audit & history"])

# ---- Dossier ----------------------------------------------------------------------------------------------------------------
with tab_d:
    b1, b2, b3 = st.columns([1, 1, 3])
    if res["pdf"]:
        b1.download_button("⬇ Dossier (PDF)", open(res["pdf"], "rb"), file_name="dossier.pdf", type="primary")
    else:
        b1.warning("PDF could not be built")
    b2.download_button("⬇ Dossier (JSON)", open(res["json"], "rb"), file_name="dossier.json")
    st.write(dj["summary"])
    st.markdown("#### Findings")
    show = st.radio("Show", ["Needs review (high and medium)", "All findings"], horizontal=True, label_visibility="collapsed")
    shown = [f for f in REAL if show.startswith("All") or f.severity != "low"]
    if not shown:
        st.info("None in this view." if REAL else "No findings. This does not prove authenticity.")
    for i, f in enumerate(sorted(shown, key=lambda x: (SEV[x.severity], -x.confidence))):
        finding_card(f, expanded=i < 2)
    st.caption(GUIDE)
    corr = res.get("correlation") or {}
    if corr.get("matrix"):
        n_diff = sum(m["verdict"] == "differ" for m in corr["matrix"])
        with st.expander(f"Cross-document comparison · {len(corr['matrix'])} fields compared, {n_diff} differ"):
            VERD = {"agree": "✅ agree", "differ": "🟧 DIFFER", "single": "one document", "unverifiable": "could not compare"}
            st.dataframe(_frame([{"field": m["field"], "result": VERD.get(m["verdict"], m["verdict"]), **{e["doc_id"]: e["raw"] for e in m["entries"]}}
                                 for m in corr["matrix"]]), hide_index=True, width="stretch")
            if corr.get("unlinked"):
                st.warning("Sharing no identifier with the other documents: " + ", ".join(corr["unlinked"]))
    ver = dj.get("verification", [])
    with st.expander(f"Areas requiring additional verification · {len(ver)}"):
        st.dataframe(_frame([{"area": a_["area"], "why": a_["detail"]} for a_ in ver]), hide_index=True, width="stretch")
    with st.expander("What was checked · confidence levels"):
        for line in dj.get("processing", []):
            st.caption("• " + line)
        st.caption(dj.get("confidence_scale", ""))

# ---- Documents --------------------------------------------------------------------------------------------------------------
with tab_docs:
    st.dataframe(_frame([{"document": d.doc_id, "type": d.doc_type, "fields": len(d.fields), "SHA-256": d.sha256[:12],
                          "flagged": ", ".join(sorted({f.fid for f in REAL for r in f.refs if r["doc_id"] == d.doc_id})) or "-"} for d in D]),
                 hide_index=True, width="stretch")
    pick = st.selectbox("Document", [d.doc_id for d in D], label_visibility="collapsed")
    d = next(x for x in D if x.doc_id == pick)
    ex = explain(d)
    left, right = st.columns([2, 3])
    left.image(res["annotated"][d.doc_id], caption="Boxes are labelled with finding IDs")
    right.markdown(f"**{d.doc_type.replace('_', ' ')}** · classification {d.type_conf} · text read by {ex['ocr_engine']} ({ex['ocr_conf']})")
    if ex.get("category"):
        right.caption(f"Semantic category: {ex['category']}")
    if d.fields:
        right.dataframe(_frame([{"field": f.key, "value": f.raw, "confidence": f.conf} for f in d.fields]), hide_index=True, width="stretch")
    else:
        right.info("No fields extracted: " + str(d.class_info.get("decision", "")))
    if d.sections or d.tables:
        with st.expander(f"Tables and sections · {len(d.tables)} table(s), {len(d.sections)} section(s)"):
            lay = d.facts.get("layout")
            if lay:
                st.caption(f"{lay['label']}. " + "; ".join(lay["evidence"]))
            for t in d.tables:
                chk = t["checks"]
                if t["kind"] == "transactions":
                    st.markdown(f"**{t['tid']} transactions** · running balance checked on {chk['checked']} rows, {len(chk['breaks'])} break(s)")
                    st.dataframe(_frame([{"#": r["n"], "date": r["date"], "description": r["desc"], "withdrawal": r["debit"], "deposit": r["credit"], "balance": r["balance"]}
                                         for r in t["rows"]]), hide_index=True, width="stretch")
                else:
                    st.markdown(f"**{t['tid']} earnings / deductions** · {len(chk['mismatches'])} total mismatch(es)")
                    st.dataframe(_frame([{"side": r["side"], "item": r["label"], "amount": r["value"]} for r in t["rows"]]), hide_index=True)
    with st.expander("How this page was processed (technical)"):
        st.markdown(f"**Classification:** {ex['decision']}")
        notes = ([{"note": "value could not be read", "field": u["key"], "text": u["raw"]} for u in d.facts.get("unparsed", [])] +
                 [{"note": f"value looks like {u['kind']}", "field": u["key"], "text": u["raw"]} for u in d.facts.get("rejected", [])] +
                 [{"note": "second, different value", "field": u["key"], "text": f"{u['kept']['raw']} vs {u['other']['raw']}"} for u in d.facts.get("duplicates", [])])
        if notes:
            st.dataframe(_frame(notes), hide_index=True)
        st.markdown("**Forensic checks:** " + "; ".join(f"{k}: {d.facts[k]}" for k in ("forensics", "typography", "grid", "photo", "signature") if d.facts.get(k)))
        if d.facts.get("pdf_checks"):
            st.caption(f"PDF structure checks: {d.facts['pdf_checks']}")
        for label, p in (("ELA heat map", os.path.join(res["outdir"], f"{d.doc_id}_ela.png")), ("Located regions", (res.get("regions") or {}).get(d.doc_id)),
                         ("Structure overlay", (res.get("structure") or {}).get(d.doc_id))):
            if p and os.path.exists(p):
                st.image(p, caption=label)
        st.markdown("**Fields as matched**")
        st.dataframe(_frame([{"field": f.key, "label read": f.label, "matched by": f.match, "score": f.label_sim, "box": str(list(f.bbox))} for f in d.fields]), hide_index=True, width="stretch")

# ---- Audit & history --------------------------------------------------------------------------------------------------------
with tab_a:
    (st.success if ai.get("chain_ok") else st.error)(f"Audit log of this run: {ai.get('events')} events, hash chain " + ("verified intact (in memory and on disk)" if ai.get("chain_ok") else "NOT VERIFIED"))
    st.caption(f"Run {ai.get('run_id')} · final digest")
    st.code(dj.get("audit_digest", ""))
    st.markdown("#### Investigation history")
    st.dataframe(_frame([{"run": h["run_id"], "finished": h.get("finished", "")[:19], "findings": h.get("findings"), "high": h.get("high"),
                          "digest": str(h.get("audit_digest", ""))[:16], "chain": "intact" if h.get("chain_ok") else "NOT VERIFIED"} for h in load_history(res["outdir"])]),
                 hide_index=True, width="stretch")
    dec = load_decisions(res["outdir"])
    st.markdown("#### Reviewer decisions")
    if dec:
        st.caption("Review log chain: " + ("intact" if verify_decisions(res["outdir"]) else "NOT VERIFIED"))
        st.dataframe(_frame([{"time": e["ts"][:19], "run": e.get("run"), "finding": e["fid"], "decision": e["decision"], "note": e.get("note", ""), "reviewer": e.get("reviewer", "")}
                             for e in dec]), hide_index=True, width="stretch")
        st.download_button("⬇ Review log (JSONL)", open(os.path.join(res["outdir"], REVIEW_LOG), "rb"), file_name="review_log.jsonl")
    else:
        st.caption("No reviewer decisions recorded yet. Record them on each finding in the Dossier tab.")
    with st.expander(f"Processing log · {len(dj['audit_trail'])} events"):
        st.dataframe(_frame(dj["audit_trail"]), hide_index=True, width="stretch")
