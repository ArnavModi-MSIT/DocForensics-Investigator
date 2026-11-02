"""Semantic page-type scoring with sentence embeddings (optional, falls back safely).

Backends tried in order: fastembed (BAAI/bge-small-en-v1.5, ONNX, no torch) ->
sentence-transformers (all-MiniLM-L6-v2). Models are cached in ./models so a demo can run offline
after one successful download:   python -m docforensics.embed_classifier --check
Env: DOCFOR_EMBED=off  (disable) | stub (TEST ONLY lexical hashing embedder) | auto (default)
"""
import os, re, sys, zlib
import numpy as np

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
SUPPORTED = ["application_form", "id_card", "salary_slip", "bank_statement"]

# Reference descriptions. "other:*" types are negative classes for unsupported pages.
PROTOTYPES = {
    "application_form": [
        "Loan application form. Applicant name, date of birth, ID number, address, employer, declared monthly income.",
        "Credit application with applicant details and a declaration of income, signed by the applicant."],
    "id_card": [
        "National identity card. Name, date of birth, ID number, address, issue date.",
        "Government issued identification card of the holder with identity number and date of issue."],
    "salary_slip": [
        "Salary slip for the month. Employee name, employer, gross earnings, total deductions, net pay.",
        "Payslip showing earnings and deductions breakdown and the net salary paid."],
    "bank_statement": [
        "Bank account statement. Account holder, account number, statement period, opening balance, salary credit, closing balance.",
        "Statement of account listing credits, debits and running balance for a period."],
    "other:research_paper": [
        "Abstract. Introduction. Related work. Methodology. Experiments and results. Conclusion. References. A research paper on machine learning."],
    "other:resume": [
        "Curriculum vitae resume. Education, technical skills, projects, internships, work experience, achievements.",
        "Student resume: B.Tech information technology, skills in Python, SQL, Power BI, projects, internship, CGPA, certifications.",
        "Professional summary and career objective. Skills, tools, projects, experience, education, contact email, LinkedIn and GitHub."],
    "other:invoice": [
        "Tax invoice with item descriptions, quantities, unit price, GST and total amount payable."],
    "other:article_letter": [
        "A letter, report or article written in paragraphs of prose with headings and no form fields."],
    "other:financial_report": [
        "Quarterly financial statement of a company. Sales, EBIT, net income, earnings per share, segment results, balance sheet, cash flow, shareholders, outlook.",
        "Annual report: management discussion and analysis, consolidated income statement, equity, dividends, investor relations, forward-looking statements."],
    "other:business_report": [
        "Sales report with regional performance, KPIs, growth versus last quarter, charts, market analysis, forecast and recommendations."],
    "other:contract": [
        "Agreement between the parties. Terms and conditions, clauses, obligations, termination, governing law, jurisdiction, signed by both parties.",
        "Terms of service, privacy policy, non-disclosure agreement, lease or employment contract."],
    "other:tax_form": [
        "Income tax return or Form 16 / TDS certificate. Assessment year, PAN of deductor and deductee, tax deducted at source, taxable income, Section 80C."],
    "other:utility_bill": [
        "Electricity, gas, water or telephone bill. Consumer number, billing period, meter reading, units consumed, amount due, due date."],
    "other:certificate": [
        "Certificate or marksheet issued by a school or university. This is to certify that the student has passed, grade, marks obtained, seat number, degree awarded."],
    "other:medical_report": [
        "Medical or laboratory report. Patient name, doctor, diagnosis, test results with reference ranges, prescription, hospital."],
    "other:presentation_brochure": [
        "Presentation slides or brochure: short headlines, bullet points, product features, company overview, contact us, marketing text."],
}

# Human-readable names for every class (supported types and the broader category list).
CATEGORY_NAMES = {
    "application_form": "Application form", "id_card": "ID card", "salary_slip": "Salary slip", "bank_statement": "Bank statement",
    "other:research_paper": "Research paper", "other:resume": "Resume / CV", "other:invoice": "Invoice / bill",
    "other:article_letter": "Letter / article / prose document", "other:financial_report": "Corporate financial report",
    "other:business_report": "Business / sales report", "other:contract": "Contract / agreement / policy",
    "other:tax_form": "Tax form / certificate", "other:utility_bill": "Utility bill", "other:certificate": "Academic certificate / marksheet",
    "other:medical_report": "Medical / lab report", "other:presentation_brochure": "Presentation / brochure",
}
CAT_TEMP = float(os.environ.get("DOCFOR_CAT_TEMP", "0.03"))   # softmax temperature turning cosine scores into a confidence


def _probs(sims):
    keys = list(sims)
    z = np.array([sims[t] for t in keys]) / CAT_TEMP
    z = np.exp(z - z.max())
    return keys, z / z.sum()


def _pack(keys, prob, cos, k=3):
    order = np.argsort(-prob)
    top = [{"key": keys[i], "name": CATEGORY_NAMES.get(keys[i], keys[i]), "confidence": round(float(prob[i]), 3),
            "cosine": round(float(cos[keys[i]]), 3)} for i in order[:k]]
    return {"top": top, "label": top[0]["name"], "confidence": top[0]["confidence"], "key": top[0]["key"],
            "margin": round(top[0]["cosine"] - (top[1]["cosine"] if len(top) > 1 else 0), 3)}


def category_from_sims(sims, k=3):
    """Turn per-class cosine similarities into a ranked category list with a confidence score.
    Confidence = softmax(sim / T) over all classes: a relative score (how clearly the best class beats the others),
    not a calibrated probability. Also returns the raw cosine and the margin over the runner-up."""
    keys, prob = _probs(sims)
    return _pack(keys, prob, sims, k)


def category_from_pages(sims_list, weights=None, k=3):
    """File-level category from several pages: average of the per-page probability vectors (weighted by text amount), so a
    few numeric table pages that look like 'nothing in particular' cannot drown the pages that clearly say what it is."""
    base = np.array(weights if weights is not None else [1.0] * len(sims_list), dtype=float)
    probs = [_probs(sm)[1] for sm in sims_list]
    w = base * np.array([p.max() for p in probs]) ** 2           # decisive pages (cover, summary) count more than ambiguous tables
    w = w / w.sum()
    keys = list(sims_list[0])
    prob = sum(wi * pi for wi, pi in zip(w, probs))
    cos = {t: float(sum(wi * sm[t] for wi, sm in zip(w, sims_list))) for t in keys}
    out = _pack(keys, prob, cos, k)
    out["pages_agree"] = [int(sum(1 for p in probs if keys[int(p.argmax())] == out["key"])), len(probs)]
    return out


def describe(cat):
    """One-line text for a category dict, e.g. 'Corporate financial report (confidence 0.73; 9 of 10 pages agree)'."""
    if not cat:
        return ""
    t = f"{cat['label']} (confidence {cat['confidence']:.2f}"
    if cat.get("pages_agree") and cat["pages_agree"][1] > 1:
        t += f"; {cat['pages_agree'][0]} of {cat['pages_agree'][1]} pages agree"
    return t + ")"


_state = {"emb": None, "loaded": False, "error": "", "proto": None}


class _Stub:
    name = "stub-hashing (TEST ONLY, lexical, not semantic)"

    def embed(self, texts):
        out = np.zeros((len(texts), 512))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, zlib.crc32(w.encode()) % 512] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class _Fast:
    def __init__(self):
        from fastembed import TextEmbedding
        self.m = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=MODEL_DIR)
        self.name = "fastembed:BAAI/bge-small-en-v1.5"

    def embed(self, texts):
        v = np.array(list(self.m.embed(texts)), dtype=float)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


class _ST:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODEL_DIR)
        self.name = "sentence-transformers:all-MiniLM-L6-v2"

    def embed(self, texts):
        v = np.array(self.m.encode(list(texts)), dtype=float)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def get_embedder():
    if _state["loaded"]:
        return _state["emb"]
    _state["loaded"] = True
    mode = os.environ.get("DOCFOR_EMBED", "auto").lower()
    errs = []
    if mode == "off":
        _state["error"] = "disabled (DOCFOR_EMBED=off)"
    elif mode == "stub":
        _state["emb"] = _Stub()
    else:
        for cls in (_Fast, _ST):
            try:
                _state["emb"] = cls()
                break
            except Exception as e:                  # missing package, no internet on first run, ...
                errs.append(f"{cls.__name__}: {type(e).__name__}: {str(e)[:90]}")
        _state["error"] = " | ".join(errs)
    return _state["emb"]


def status():
    e = get_embedder()
    return {"active": e is not None, "backend": e.name if e else None, "detail": _state["error"]}


def semantic_scores(text):
    """Return {'backend', 'sims': {type: max cosine}} or None when no embedder is available."""
    e = get_embedder()
    if e is None or not text.strip():
        return None
    try:
        if _state["proto"] is None:
            flat = [(t, s) for t, ss in PROTOTYPES.items() for s in ss]
            _state["proto"] = ([t for t, _ in flat], e.embed([s for _, s in flat]))
        types, P = _state["proto"]
        v = e.embed([text[:2000]])[0]
        sims = {}
        for t, s in zip(types, P @ v):
            sims[t] = max(sims.get(t, -1.0), float(s))
        return {"backend": e.name, "sims": {k: round(x, 3) for k, x in sims.items()}}
    except Exception as ex:
        _state["error"] = f"runtime: {type(ex).__name__}: {str(ex)[:90]}"
        return None


if __name__ == "__main__":
    st = status()
    print("embedding backend:", st)
    if "--check" in sys.argv:
        if not st["active"]:
            print("No embedding model available. Install: pip install fastembed   (needs internet once to download ~70MB)")
            sys.exit(1)
        tests = {
            "salary slip": "SALARY SLIP Employee Name: Rohan Mehta Employer: Nimbus Pay Month: October Gross Earnings: Rs. 95,000 Total Deductions Net Pay",
            "loan application": "LOAN APPLICATION FORM Applicant Name Date of Birth ID Number Address Employer Declared Monthly Income",
            "research paper": "Abstract We present an applicant-facing job recommendation prototype. Results show improved precision. References",
            "resume": "Rahul Mehta B.Tech IT Student SQL Python Power BI Education Technical Skills Projects Internship",
        }
        for name, txt in tests.items():
            r = semantic_scores(txt)
            top = sorted(r["sims"].items(), key=lambda kv: -kv[1])[:3]
            print(f"{name:18} ->", ", ".join(f"{k} {v:.2f}" for k, v in top))
