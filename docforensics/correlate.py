"""Contextual verification (5.5) and cross-document correlation (5.6).
Every finding carries refs back to (doc, field, bbox) so it is traceable to the source."""
import re
from datetime import date
from rapidfuzz import fuzz
from .schema import Finding

REQUIRED = {
    "application_form": ["name", "dob", "id_number", "declared_income"],
    "id_card": ["name", "dob", "id_number"],
    "salary_slip": ["name", "gross", "deductions", "net_pay"],
    "bank_statement": ["name", "account_no", "salary_credit"],
}
PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")


def _ref(doc, key):
    f = doc.get(key)
    return {"doc_id": doc.doc_id, "field": key, "value": f.raw, "bbox": list(f.bbox)}


class _Ids:
    def __init__(self, prefix):
        self.p, self.n = prefix, 0

    def __call__(self):
        self.n += 1
        return f"{self.p}-{self.n:03d}"


def _norm_text(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def check_intra(docs):
    out, nid = [], _Ids("C")
    unknown = [d for d in docs if d.doc_type == "unknown"]
    if unknown:                               # report once; do not flag fields on unsupported pages
        out.append(Finding(nid(), "missing", f"{len(unknown)} page(s) not recognised as a supported document type",
                           "low", 0.7, "observed",
                           "These pages did not match any supported type (application form, ID card, salary slip, bank "
                           "statement), so field-level checks were skipped for them: "
                           + "; ".join(f"{d.doc_id} ({d.class_info.get('decision', 'unrecognised')})" for d in unknown[:6])
                           + (" ..." if len(unknown) > 6 else ""),
                           [{"doc_id": d.doc_id, "field": None, "value": None, "bbox": None} for d in unknown[:12]],
                           method="document classification"))
    for d in docs:
        if d in unknown:
            continue
        # missing required fields
        for key in REQUIRED.get(d.doc_type, []):
            if not d.get(key):
                out.append(Finding(nid(), "missing", f"Required field '{key}' not found in {d.doc_id}", "low", 0.6,
                                   "observed", f"Expected field '{key}' for a {d.doc_type} was not extracted "
                                   f"(absent, illegible or OCR failure). Verify manually.", [{"doc_id": d.doc_id, "field": key, "value": None, "bbox": None}],
                                   method="schema completeness"))
        # PAN-like format
        f = d.get("id_number")
        if f and not PAN_RE.match(str(f.value)):
            out.append(Finding(nid(), "intra", f"ID number format invalid in {d.doc_id}", "medium", 0.7, "inconsistency",
                               f"'{f.raw}' does not match the expected 5 letters + 4 digits + 1 letter pattern.",
                               [_ref(d, "id_number")], method="format validation"))
        # plausible age / dates
        dob, iss = d.get("dob"), d.get("issue_date")
        if dob and re.match(r"\d{4}-\d{2}-\d{2}", str(dob.value)):
            y, m, dd = map(int, str(dob.value).split("-"))
            age = (date.today() - date(y, m, dd)).days / 365.25
            if not 18 <= age <= 100:
                out.append(Finding(nid(), "intra", f"Implausible applicant age in {d.doc_id}", "medium", 0.7, "inconsistency",
                                   f"Date of birth {dob.raw} implies age {age:.0f}.", [_ref(d, "dob")], method="plausibility"))
            if iss and re.match(r"\d{4}-\d{2}-\d{2}", str(iss.value)) and str(iss.value) < str(dob.value):
                out.append(Finding(nid(), "intra", f"Issue date precedes date of birth in {d.doc_id}", "high", 0.9, "inconsistency",
                                   "The document claims to be issued before the holder was born.",
                                   [_ref(d, "dob"), _ref(d, "issue_date")], method="date logic"))
        # salary slip arithmetic
        if d.doc_type == "salary_slip":
            g, de, n = d.get("gross"), d.get("deductions"), d.get("net_pay")
            if g and de and n and all(isinstance(x.value, float) for x in (g, de, n)):
                exp = g.value - de.value
                if abs(exp - n.value) > 1:
                    out.append(Finding(nid(), "intra", f"Salary slip arithmetic does not balance in {d.doc_id}", "high", 0.95,
                                       "inconsistency",
                                       f"Gross {g.value:,.0f} - deductions {de.value:,.0f} = {exp:,.0f}, but Net Pay shows {n.value:,.0f} "
                                       f"(difference {n.value-exp:+,.0f}).", [_ref(d, "gross"), _ref(d, "deductions"), _ref(d, "net_pay")],
                                       method="arithmetic check"))
    return out


def _pairs(docs, key):
    have = [d for d in docs if d.get(key)]
    return [(a, b) for i, a in enumerate(have) for b in have[i + 1:]]


def check_cross(docs):
    out, nid = [], _Ids("X")
    # identity-like fields
    spec = {"name": ("Applicant name", "text"), "dob": ("Date of birth", "exact"),
            "id_number": ("ID number", "exact"), "address": ("Address", "text"),
            "employer": ("Employer", "text"), "account_no": ("Account number", "exact")}
    for key, (label, mode) in spec.items():
        have = [d for d in docs if d.get(key)]
        if len(have) < 2:
            continue

        def same(x, y):
            if mode == "exact":
                return str(x) == str(y)
            return fuzz.token_sort_ratio(_norm_text(x), _norm_text(y)) >= 99
        groups = []                                   # cluster documents by agreeing value
        for d in have:
            for g in groups:
                if same(g[0].get(key).value, d.get(key).value):
                    g.append(d)
                    break
            else:
                groups.append([d])
        if len(groups) == 1:
            continue
        groups.sort(key=len, reverse=True)
        ref = groups[0]
        majority = len(ref) > len(groups[1])          # a clear majority identifies the odd-one-out
        pairs = [(ref, g) for g in groups[1:]] if majority else [(groups[0], groups[1])]
        for rg, og in pairs:
            va, vb = rg[0].get(key).value, og[0].get(key).value
            sim = fuzz.ratio(str(va), str(vb)) if mode == "exact" else fuzz.token_sort_ratio(_norm_text(va), _norm_text(vb))
            near = sim >= 80
            sev = "high" if key in ("name", "dob", "id_number") else "medium"
            note = ("Values differ only slightly (similarity %d%%): possible typo, transliteration or minor alteration." % sim
                    if near else "Values are clearly different (similarity %d%%)." % sim)
            who = ", ".join(d.doc_id for d in og)
            agree = ", ".join(f"{d.doc_id}: {d.get(key).raw!r}" for d in rg)
            title = (f"{label} in {who} disagrees with {len(rg)} other document(s)" if majority
                     else f"{label} mismatch: {rg[0].doc_id} vs {who}")
            out.append(Finding(nid(), "cross", title, sev, 0.9 if mode == "exact" else 0.8, "inconsistency",
                               f"{who}: {og[0].get(key).raw!r} vs [{agree}]. {note}"
                               + ("" if majority else " No majority: cannot tell which document is correct."),
                               [_ref(d, key) for d in og] + [_ref(d, key) for d in rg],
                               metrics={"similarity": sim, "odd_one_out": who if majority else None},
                               method=f"cross-document {mode} match"))
    # income consistency: declared vs payslip net vs bank salary credit
    by = {d.doc_type: d for d in docs}
    form, slip, bank = by.get("application_form"), by.get("salary_slip"), by.get("bank_statement")
    evid = []
    if slip and slip.get("net_pay") and isinstance(slip.get("net_pay").value, float):
        evid.append((slip, "net_pay", slip.get("net_pay").value))
    if bank and bank.get("salary_credit") and isinstance(bank.get("salary_credit").value, float):
        evid.append((bank, "salary_credit", bank.get("salary_credit").value))
    if len(evid) == 2 and abs(evid[0][2] - evid[1][2]) > 0.01 * max(evid[0][2], evid[1][2]):
        out.append(Finding(nid(), "cross", "Salary slip net pay does not match bank salary credit", "high", 0.9, "inconsistency",
                           f"Net pay on slip {evid[0][2]:,.0f} vs salary credited in bank statement {evid[1][2]:,.0f}.",
                           [_ref(d, k) for d, k, _ in evid], method="income reconciliation"))
    if form and form.get("declared_income") and isinstance(form.get("declared_income").value, float):
        decl = form.get("declared_income").value
        for d, k, v in evid:
            if decl > 1.05 * v:
                out.append(Finding(nid(), "cross", f"Declared income exceeds income shown in {d.doc_id}", "high", 0.85, "inconsistency",
                                   f"Declared monthly income {decl:,.0f} vs {d.doc_id} {k.replace('_', ' ')} {v:,.0f} "
                                   f"({(decl/v-1)*100:.0f}% higher).",
                                   [_ref(form, "declared_income"), _ref(d, k)], method="income reconciliation"))
    return out


def corroborate(visual, others):
    """If a content inconsistency touches a field that also shows visual tampering evidence,
    raise severity/confidence: two independent lines of evidence."""
    vis = {(r["doc_id"], r["field"]): v for v in visual for r in v.refs}
    for o in others:
        hit = [vis[(r["doc_id"], r["field"])] for r in o.refs if (r["doc_id"], r["field"]) in vis]
        if hit:
            o.severity = "high"
            o.confidence = round(min(0.98, o.confidence + 0.05), 2)
            o.explanation += f" Corroborated by visual evidence ({', '.join(h.fid for h in hit)}) on the same field."
            o.metrics["corroborated_by"] = [h.fid for h in hit]
            for h in hit:
                h.confidence = round(min(0.98, h.confidence + 0.03), 2)
                h.metrics.setdefault("corroborated_by", []).append(o.fid)
    return others
