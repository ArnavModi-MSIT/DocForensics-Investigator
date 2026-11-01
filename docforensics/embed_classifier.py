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
        "Curriculum vitae resume. Education, technical skills, projects, internships, work experience, achievements."],
    "other:invoice": [
        "Tax invoice with item descriptions, quantities, unit price, GST and total amount payable."],
    "other:article_letter": [
        "A letter, report or article written in paragraphs of prose with headings and no form fields."],
}

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
            print("No embedding model available. Install: pip install fastembed   (needs internet once to download ~130MB)")
            sys.exit(1)
        tests = {
            "salary slip": "SALARY SLIP Employee Name: Rohan Mehta Employer: Nimbus Pay Month: October Gross Earnings: Rs. 95,000 Total Deductions Net Pay",
            "loan application": "LOAN APPLICATION FORM Applicant Name Date of Birth ID Number Address Employer Declared Monthly Income",
            "research paper": "Abstract We present an applicant-facing job recommendation prototype. Results show improved precision. References",
            "resume": "Arnav Modi B.Tech IT Student SQL Python Power BI Education Technical Skills Projects Internship",
        }
        for name, txt in tests.items():
            r = semantic_scores(txt)
            top = sorted(r["sims"].items(), key=lambda kv: -kv[1])[:3]
            print(f"{name:18} ->", ", ".join(f"{k} {v:.2f}" for k, v in top))
