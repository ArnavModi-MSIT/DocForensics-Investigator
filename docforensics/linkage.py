"""Cross-document correlation (5.6), part 2: which documents belong together, and what was compared.

check_cross() reports DISCREPANCIES. This module adds the rest of the picture:
  case codes   a reference printed in a page header ('FDX-2026-014/APP') or as a field must be the same on every page of one application
  links        two documents are linked when they share an agreeing identifier (ID number, employee ID, account number, case code, name, date of birth ...)
  unlinked     a document that shares nothing with any other document may belong to a different application
  matrix       for every compared field: the value in each document and whether they agree, differ, or could not be compared
Everything keeps (document, field, value, box) so each cell traces back to its source. Nothing here decides which document is right.
"""
import re
from .schema import Finding
from .compare import compare
from .pdf_layer import widen

CASE_CODE_RE = re.compile(r"^([A-Za-z]{2,6}-\d{4}-\d{2,6})(?:/[A-Za-z]{2,6})?$")
ENTITY_KEYS = [("name", "Name"), ("dob", "Date of birth"), ("father_name", "Father's name"), ("id_number", "ID number"),
               ("aadhaar_no", "Aadhaar number"), ("epic_no", "Voter ID (EPIC)"), ("address", "Address"), ("employer", "Employer"),
               ("employee_id", "Employee ID"), ("account_no", "Account number")]
LINK_KEYS = {"id_number", "aadhaar_no", "epic_no", "employee_id", "account_no", "case_code", "name", "dob"}


def _bb(doc, box):
    if not box:
        return None
    return widen(box) if doc.born_digital and doc.pdf else list(box)


def case_code(doc):
    """(code, raw, bbox) from a case-reference field or a header chip such as 'FDX-2026-014/ID'; None when the page has none."""
    f = doc.get("case_ref")
    if f:
        m = CASE_CODE_RE.match(str(f.raw).strip())
        if m:
            return m.group(1).upper(), f.raw, list(f.bbox)
    for w in doc.words:
        m = CASE_CODE_RE.match(w.text.strip())
        if m and "/" in w.text and w.bbox[1] < 0.2 * doc.height:                  # header chip only: 'CODE/SUFFIX' in the top fifth
            return m.group(1).upper(), w.text, _bb(doc, w.bbox)
    return None


def _entry(doc, key):
    f = doc.get(key)
    return None if not f else {"doc_id": doc.doc_id, "raw": f.raw, "value": f.value, "bbox": list(f.bbox), "field": key}


def _verdict(key, entries):
    if len(entries) < 2:
        return "single"
    groups, unver = [], False
    for e in entries:
        for g in groups:
            r = compare(key, g[0]["value"], e["value"])
            if r.verdict in ("same", "compatible"):
                g.append(e)
                break
            if r.verdict == "unverifiable":
                unver = True
                g.append(e)
                break
        else:
            groups.append([e])
    return "differ" if len(groups) > 1 else ("unverifiable" if unver else "agree")


def _amount_verdict(vals):
    if len(vals) < 2:
        return "single"
    hi, lo = max(vals), min(vals)
    return "agree" if hi - lo <= max(1.0, 0.01 * hi) else "differ"


def _num(d, key):
    f = d.get(key)
    return f.value if f and isinstance(f.value, float) else None


def build_correlation(docs):
    """{'matrix': [...], 'links': [...], 'unlinked': [...], 'case_codes': {...}} over the recognised pages (continuation pages count as their file)."""
    rec = [d for d in docs if d.doc_type != "unknown" and not d.class_info.get("continuation_of")]
    matrix = []
    for key, label in ENTITY_KEYS:
        es = [e for e in (_entry(d, key) for d in rec) if e]
        if es:
            matrix.append({"field": label, "key": key, "verdict": _verdict(key, es), "entries": es})
    codes = {d.doc_id: case_code(d) for d in rec}
    ce = [{"doc_id": i, "raw": c[1], "value": c[0], "bbox": c[2], "field": "case_ref"} for i, c in codes.items() if c]
    if ce:
        matrix.append({"field": "Case reference", "key": "case_code", "verdict": "single" if len(ce) < 2 else ("agree" if len({e["value"] for e in ce}) == 1 else "differ"),
                       "entries": ce})
    for label, srcs in (("Monthly gross income", (("application_form", "declared_income"), ("employment_letter", "monthly_gross"), ("salary_slip", "gross"))),
                        ("Annual gross income", (("application_form", "annual_gross"), ("employment_letter", "annual_gross")))):
        es = []
        for d in rec:
            for t, k in srcs:
                f = d.get(k)
                if d.doc_type == t and f and isinstance(f.value, float) and (k != "declared_income" or "gross" in f.label.lower()):
                    es.append({"doc_id": d.doc_id, "raw": f.raw, "value": f.value, "bbox": list(f.bbox), "field": k})
        if es:
            matrix.append({"field": label, "key": "amount", "verdict": _amount_verdict([e["value"] for e in es]), "entries": es})
    pm = [e for e in (_entry(d, "pay_month") for d in rec) if e]
    if pm:
        norm = {re.sub(r"[^a-z0-9]", "", str(e["raw"]).lower()) for e in pm}
        matrix.append({"field": "Pay month", "key": "pay_month", "verdict": "single" if len(pm) < 2 else ("agree" if len(norm) == 1 else "differ"), "entries": pm})
    slips = [(d, _num(d, "net_pay")) for d in rec if d.doc_type == "salary_slip" and _num(d, "net_pay") is not None]
    credits = [(d, c) for d in rec if d.doc_type == "bank_statement" for c in d.facts.get("payroll_credits", [])]
    if slips and credits:
        es = [{"doc_id": d.doc_id, "raw": d.get("net_pay").raw, "value": v, "bbox": list(d.get("net_pay").bbox), "field": "net_pay"} for d, v in slips]
        es += [{"doc_id": d.doc_id, "raw": c.get("raw") or f"{c['amount']:,.0f}", "value": c["amount"], "bbox": _bb(d, c["bbox"]), "field": "payroll_credit"} for d, c in credits]
        ok = all(any(abs(v - c["amount"]) <= 0.01 * max(v, c["amount"], 1) for _, c in credits) for _, v in slips)
        matrix.append({"field": "Payslip net pay vs bank payroll credit", "key": "amount", "verdict": "agree" if ok else "differ", "entries": es})
    links = []                                                                      # pairs of documents with the identifiers they share
    for i, a in enumerate(rec):
        for b in rec[i + 1:]:
            shared = []
            for key in LINK_KEYS:
                if key == "case_code":
                    ca, cb = codes.get(a.doc_id), codes.get(b.doc_id)
                    if ca and cb and ca[0] == cb[0]:
                        shared.append(key)
                    continue
                fa, fb = a.get(key), b.get(key)
                if fa and fb and compare(key, fa.value, fb.value).verdict in ("same", "compatible"):
                    shared.append(key)
            if shared:
                links.append({"a": a.doc_id, "b": b.doc_id, "shared": sorted(shared)})
    linked = {x for l in links for x in (l["a"], l["b"])}
    unlinked = [d.doc_id for d in rec if d.doc_id not in linked] if len(rec) >= 2 and links else []
    for m in matrix:                                                                # comparison done: keep only what is safe to export (Aadhaar digits stay masked in `raw`)
        for e in m["entries"]:
            e.pop("value", None)
    return {"matrix": matrix, "links": links, "unlinked": unlinked, "case_codes": {i: c[0] for i, c in codes.items() if c},
            "documents": [d.doc_id for d in rec]}


def linkage_findings(docs, corr):
    """Findings from the linkage picture (ids R-nnn). Case codes that differ; documents that share no identifier with the others."""
    out, n = [], 0
    by_id = {d.doc_id: d for d in docs}
    row = next((m for m in corr["matrix"] if m["key"] == "case_code"), None)
    if row and row["verdict"] == "differ":
        n += 1
        vals = ", ".join(f"{e['doc_id']}: {e['raw']!r}" for e in row["entries"])
        out.append(Finding(f"R-{n:03d}", "cross", "Case reference differs between documents", "medium", 0.75, "inconsistency",
                           f"The documents of one application should carry the same case reference, but they show {vals}. A document from another "
                           "application, or an altered reference, would look like this. Check that every document belongs to this case.",
                           [{"doc_id": e["doc_id"], "field": e["field"], "value": e["raw"], "bbox": e["bbox"]} for e in row["entries"]],
                           method="case reference comparison"))
    for did in corr["unlinked"]:
        n += 1
        d = by_id[did]
        weak = sum(1 for k in LINK_KEYS if k != "case_code" and d.get(k)) < 2            # a page with hardly any identifiers cannot be linked either way
        out.append(Finding(f"R-{n:03d}", "cross", f"{did} shares no identifier with the other documents", "low" if weak else "medium", 0.4 if weak else 0.6, "observed",
                           f"None of the identifiers that tie documents to one person or application (name, date of birth, ID number, employee ID, account "
                           f"number, case reference) agree between {did} and any other document of this case, although the other documents are linked "
                           f"to each other. It may belong to a different applicant or application; verify before relying on it.",
                           [{"doc_id": did, "field": None, "value": None, "bbox": None}], method="document linkage"))
    return out
