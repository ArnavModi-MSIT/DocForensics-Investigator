"""DocForensics - Streamlit UI.   Run:  streamlit run app.py"""
import glob, os, tempfile, time
import pandas as pd
import streamlit as st
from docforensics.pipeline import run_case
from docforensics.debug import explain

st.set_page_config(page_title="DocForensics", layout="wide")
st.title("DocForensics - Document Forensics & Contextual Intelligence")
st.caption("Visual findings are indicators of possible alteration, not proof. Verify against originals.")

ROOT = os.path.dirname(__file__)
with st.sidebar:
    st.header("Case input")
    mode = st.radio("Source", ["Sample case", "Upload documents"])
    paths = []
    if mode == "Sample case":
        cases = sorted(d for d in glob.glob(os.path.join(ROOT, "samples", "case_*")))
        pick = st.selectbox("Sample", cases, format_func=os.path.basename)
        paths = sorted(glob.glob(os.path.join(pick, "*.jpg")))
        case_id = os.path.basename(pick)
    else:
        ups = st.file_uploader("Images / PDFs / ZIP (one zip = one case)", accept_multiple_files=True,
                               type=["jpg", "jpeg", "png", "pdf", "tif", "tiff", "zip"])
        case_id = st.text_input("Case ID", f"case_{int(time.time())}")
        tmp = tempfile.mkdtemp()
        for u in ups or []:
            p = os.path.join(tmp, u.name); open(p, "wb").write(u.getbuffer()); paths.append(p)
    go = st.button("Run investigation", type="primary", disabled=not paths)
    st.markdown(f"LLM summary: **{'on' if os.environ.get('ANTHROPIC_API_KEY') else 'off (template)'}**")

if go:
    bar, msg = st.progress(0.0), st.empty()
    steps = {"n": 0}
    total = 3 * len(paths) + 4

    def prog(m):
        steps["n"] += 1
        bar.progress(min(1.0, steps["n"] / total)); msg.write(m)
    try:
        st.session_state["res"] = run_case(paths, os.path.join(ROOT, "output", case_id), case_id, prog)
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
c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents", len(D)); c2.metric("Findings", len(F))
c3.metric("High severity", sum(f.severity == "high" for f in F))
c4.metric("Potential manipulation", sum(f.status == "potential_manipulation" for f in F))
st.subheader("Investigator summary")
st.write(res["dossier"]["summary"])

t1, t2, t3, t4, t5 = st.tabs(["Findings", "Documents & evidence", "Forensic signals", "Audit trail & export", "Classification & OCR debug"])
with t1:
    if not F:
        st.success("No inconsistencies or visual anomalies detected (this does not prove authenticity).")
    icon = {"potential_manipulation": "🟥", "inconsistency": "🟧", "observed": "🟦"}
    for f in sorted(F, key=lambda x: ({"high": 0, "medium": 1, "low": 2}[x.severity], -x.confidence)):
        with st.expander(f"{icon.get(f.status, '')} {f.fid} | {f.severity.upper()} | conf {f.confidence:.2f} | {f.title}"):
            st.write(f.explanation)
            st.caption(f"Status: {f.status.replace('_', ' ')}  |  Method: {f.method}")
            st.dataframe(pd.DataFrame([{k: r[k] for k in ("doc_id", "field", "value")} for r in f.refs]), hide_index=True)
with t2:
    cols = st.columns(2)
    for i, d in enumerate(D):
        with cols[i % 2]:
            st.markdown(f"**{d.doc_id}** - {d.doc_type} (classification {d.type_conf:.2f})")
            st.image(res["annotated"][d.doc_id], use_container_width=True)
            with st.expander("Extracted fields"):
                st.dataframe(pd.DataFrame([{"field": f.key, "value": f.raw, "ocr_conf": f.conf} for f in d.fields]), hide_index=True)
            with st.expander("ELA heat-map (compression error level)"):
                st.image(os.path.join(res["outdir"], f"{d.doc_id}_ela.png"), use_container_width=True)
with t3:
    st.caption("Robust z-scores of each field vs the other fields of the same page (|z| >= 4 is notable).")
    st.dataframe(pd.DataFrame(res["forensic_table"]).round(1), hide_index=True)
with t4:
    st.dataframe(pd.DataFrame(res["dossier"]["audit_trail"]), hide_index=True)
    st.download_button("Download dossier (PDF)", open(res["pdf"], "rb"), file_name="dossier.pdf")
    st.download_button("Download dossier (JSON)", open(res["json"], "rb"), file_name="dossier.json")

with t5:
    ex = {d.doc_id: explain(d) for d in D}
    emb = next(iter(ex.values()))["embedding"]
    if emb["active"]:
        st.success(f"Semantic classifier ACTIVE: {emb['backend']}")
    else:
        st.warning("Semantic classifier OFF - using keywords + structural check only. "
                   + (emb["detail"] or "") + "  (install: pip install fastembed; needs internet once)")
    st.caption("A page is a supported document only if (1) the candidate type wins on keywords/semantics, (2) it is not vetoed as a "
               "known unsupported class (research paper, resume, invoice...), and (3) at least 3 of that type's expected fields were extracted.")
    st.dataframe(pd.DataFrame([{
        "document": k, "predicted type": v["predicted"], "best guess": v["best_guess"], "evidence score": v["confidence"],
        "recognised": v["recognised"], "words read": v["n_words"], "fields extracted": v["n_fields"], "why": v["decision"]}
        for k, v in ex.items()]), hide_index=True)
    for d in D:
        v = ex[d.doc_id]
        with st.expander(f"{d.doc_id}: {v['predicted']} (best guess {v['best_guess']}, evidence {v['confidence']})"):
            st.markdown(f"**Decision:** {v['decision']}")
            st.markdown("**Evidence per supported type**")
            st.dataframe(pd.DataFrame(v["types"]), hide_index=True)
            if v["other_classes"]:
                st.markdown("**Similarity to unsupported classes (veto)**")
                st.dataframe(pd.DataFrame(v["other_classes"]), hide_index=True)
            st.markdown("**OCR text by line** (field = extracted field; matched_by = fuzzy dictionary or semantic embedding; label_score = similarity)")
            st.dataframe(pd.DataFrame(v["lines"]), hide_index=True, use_container_width=True)
