"""Contextual verification (5.5) and cross-document correlation (5.6).
Every finding carries refs back to (doc, field, bbox) so it is traceable to the source."""
import os, re
from datetime import date
from rapidfuzz import fuzz
from .schema import Finding
from .compare import compare

REQUIRED = {
    "application_form": ["name", "dob", "id_number", "declared_income"],
    "id_card": [],
    "salary_slip": ["name", "gross", "deductions", "net_pay"],
    "bank_statement": ["name", "account_no"],
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


def _cat_sentence(unknown):
    """'The file(s) most resemble X (confidence ..)' from the semantic category, when the model was available."""
    from .embed_classifier import describe
    seen, parts = set(), []
    for d in unknown:
        cat = d.class_info.get("file_category") or d.class_info.get("category")
        key = (os.path.basename(d.path), cat["label"] if cat else None)
        if cat and key not in seen:
            seen.add(key)
            parts.append(f"{os.path.basename(d.path)} most resembles: {describe(cat)}")
    return ("Semantic category: " + "; ".join(parts[:4]) + ". ") if parts else ""


def check_intra(docs):
    out, nid = [], _Ids("C")
    unknown = [d for d in docs if d.doc_type == "unknown"]
    if unknown:                               # report once; do not flag fields on unsupported pages
        out.append(Finding(nid(), "missing", f"{len(unknown)} page(s) not recognised as a supported document type",
                           "low", 0.7, "observed",
                           "These pages did not match any supported type (application form, ID card, salary slip, bank "
                           "statement), so field-level checks were skipped for them. "
                           + _cat_sentence(unknown)
                           + "Details: "
                           + "; ".join(f"{d.doc_id} ({d.class_info.get('decision', 'unrecognised')})" for d in unknown[:6])
                           + (" ..." if len(unknown) > 6 else ""),
                           [{"doc_id": d.doc_id, "field": None, "value": None, "bbox": None} for d in unknown[:12]],
                           method="document classification"))
    for d in docs:
        if d in unknown or d.class_info.get("continuation_of"):
            continue
        # missing required fields
        for key in REQUIRED.get(d.doc_type, []):
            if not d.get(key):
                out.append(Finding(nid(), "missing", f"Required field '{key}' not found in {d.doc_id}", "low", 0.6,
                                   "observed", f"Expected field '{key}' for a {d.doc_type} was not extracted "
                                   f"(absent, illegible or OCR failure). Verify manually.", [{"doc_id": d.doc_id, "field": key, "value": None, "bbox": None}],
                                   method="schema completeness"))
        seen_u = set()
        for u in d.facts.get("unparsed", []):
            if u["key"] in seen_u or d.get(u["key"]):          # one report per field; skip if another pass read it
                continue
            seen_u.add(u["key"])
            out.append(Finding(nid(), "missing", f"Value of '{u['key']}' could not be read in {d.doc_id}", "low", 0.7, "observed",
                               f"The label was found but its value ({u['raw']!r}) is not a valid {u['key'].replace('_', ' ')}. Checks that need it "
                               f"(arithmetic, cross-document comparison) were skipped for this field. Verify manually.",
                               [{"doc_id": d.doc_id, "field": u["key"], "value": u["raw"], "bbox": None}], method="value validation"))
        # Aadhaar number checksum (Verhoeff): a failing number cannot be a genuine Aadhaar
        if d.get("aadhaar_no") and d.facts.get("aadhaar_checksum_ok") is False:
            out.append(Finding(nid(), "intra", f"Aadhaar number fails its checksum in {d.doc_id}", "high", 0.9, "inconsistency",
                               "The 12-digit number does not satisfy the Verhoeff check digit that every issued Aadhaar number carries "
                               "(or starts with 0/1). It is mistyped, misread by OCR, or fabricated. Verify with the original card.",
                               [_ref(d, "aadhaar_no")], method="Verhoeff checksum"))
        # PAN-like format
        f = d.get("id_number")
        if f and (re.match(r"[A-Z]{5}", str(f.value)) or "pan" in f.label.lower()) and not PAN_RE.match(str(f.value)):
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


_DOWN = {"high": "medium", "medium": "low", "low": "low"}
_LABELS = {"name": "Applicant name", "dob": "Date of birth", "id_number": "ID number", "address": "Address",
           "employer": "Employer", "account_no": "Account number", "aadhaar_no": "Aadhaar number",
           "epic_no": "Voter ID (EPIC) number", "father_name": "Father's name"}
_EXACT_KEYS = {"dob", "id_number", "account_no", "aadhaar_no", "epic_no"}


def _min_conf(docs_, key):
    return min((d.get(key).conf for d in docs_ if d.get(key)), default=1.0)


def check_cross(docs):
    out, nid = [], _Ids("X")
    unverifiable = {}                                  # key -> [(doc_a, doc_b, why)] for coverage reporting
    for key, label in _LABELS.items():
        have = [d for d in docs if d.get(key)]
        if len(have) < 2:
            continue
        groups = []                                   # cluster documents whose values are the same or compatible
        for d in have:
            for g in groups:
                r = compare(key, g[0].get(key).value, d.get(key).value)
                if r.verdict in ("same", "compatible"):
                    g.append(d)
                    break
                if r.verdict == "unverifiable":       # cannot tell: do not raise a finding, remember it for coverage
                    unverifiable.setdefault(key, []).append((g[0].doc_id, d.doc_id, r.note))
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
            r = compare(key, va, vb)
            exact_kind = key in _EXACT_KEYS
            sev = "high" if key in ("name", "dob", "id_number") else "medium"
            if r.verdict == "near" and not exact_kind:
                sev = "medium"                         # a text near-miss is more often an OCR slip than an alteration
            weak = min(_min_conf(og, key), _min_conf(rg, key)) < 0.6
            if weak:
                sev = _DOWN[sev]
            note = ("Values differ only slightly (similarity %d%%): possible OCR slip, typo, transliteration or minor alteration." % r.sim
                    if r.verdict == "near" else "Values are clearly different" + (f" ({r.note})" if r.note else f" (similarity {r.sim}%)") + ".")
            if weak:
                note += " OCR confidence on at least one value is below 0.60, so this may be a misread: severity lowered, verify manually."
            who = ", ".join(d.doc_id for d in og)
            agree = ", ".join(f"{d.doc_id}: {d.get(key).raw!r}" for d in rg)
            title = (f"{label} in {who} disagrees with {len(rg)} other document(s)" if majority
                     else f"{label} mismatch: {rg[0].doc_id} vs {who}")
            out.append(Finding(nid(), "cross", title, sev, 0.9 if exact_kind else 0.8, "inconsistency",
                               f"{who}: {og[0].get(key).raw!r} vs [{agree}]. {note}"
                               + ("" if majority else " No majority: cannot tell which document is correct."),
                               [_ref(d, key) for d in og] + [_ref(d, key) for d in rg],
                               metrics={"similarity": r.sim, "verdict": r.verdict, "odd_one_out": who if majority else None},
                               method=f"cross-document {key} comparison"))
    if unverifiable:
        det = "; ".join(f"{_LABELS[k]}: {len(v)} pair(s) ({v[0][2]})" for k, v in unverifiable.items())
        out.append(Finding("N-004", "missing", "Some cross-document comparisons were not possible", "low", 0.9, "observed",
                           "The values were too short or incomplete to compare reliably, so no mismatch was raised and none was ruled out. " + det,
                           [{"doc_id": v[0][0], "field": k, "value": None, "bbox": None} for k, v in list(unverifiable.items())[:6]],
                           method="cross-document coverage"))
    out.extend(_income_checks(docs, nid))
    return out


def _num(d, key):
    f = d.get(key)
    return f.value if f and isinstance(f.value, float) else None


def _income_checks(docs, nid):
    """Declared income vs payslips vs bank credits, across ALL payslips / statement pages (not one per type).
    People often declare GROSS monthly income while the bank shows NET credit, so a declared amount is only flagged
    HIGH when it exceeds even the largest amount any document shows; against net-only evidence it is flagged as a
    medium 'verify' item that names the gross/net possibility."""
    out = []
    slips = [(d, _num(d, "net_pay"), _num(d, "gross")) for d in docs if d.doc_type == "salary_slip" and _num(d, "net_pay") is not None]
    credits = [(d, _num(d, "salary_credit")) for d in docs if d.doc_type == "bank_statement" and _num(d, "salary_credit") is not None]
    forms = [d for d in docs if d.doc_type == "application_form" and _num(d, "declared_income") is not None]
    for sd, net, _gross in slips:                                   # each payslip must be matched by some bank credit
        if credits and not any(abs(net - cv) <= 0.01 * max(abs(net), abs(cv), 1.0) for _, cv in credits):
            cd, cv = min(credits, key=lambda x: abs(x[1] - net))
            out.append(Finding(nid(), "cross", "Salary slip net pay does not match bank salary credit", "high", 0.9, "inconsistency",
                               f"Net pay on {sd.doc_id} {net:,.0f} vs closest salary credit in the bank statement ({cd.doc_id}) {cv:,.0f}.",
                               [_ref(sd, "net_pay"), _ref(cd, "salary_credit")], method="income reconciliation"))
    for fd in forms:
        decl = _num(fd, "declared_income")
        # A payslip the bank contradicts cannot vouch for a declared amount (a forger can raise both together), so when
        # bank credits exist only payslips whose net pay matches a credit count as supporting evidence.
        trusted = [t for t in slips if not credits or any(abs(t[1] - cv) <= 0.01 * max(abs(t[1]), abs(cv), 1.0) for _, cv in credits)]
        amounts = [(sd, "gross", g) for sd, _n, g in trusted if g is not None] + [(sd, "net_pay", n) for sd, n, _g in trusted] + \
                  [(cd, "salary_credit", cv) for cd, cv in credits]
        if not amounts:
            continue
        top = max(amounts, key=lambda x: x[2])
        has_gross = any(k == "gross" for _, k, _ in amounts)
        for d, k, v in sorted(amounts, key=lambda x: -x[2])[:1]:
            if decl > 1.05 * v:
                pct = f"{(decl / v - 1) * 100:.0f}% higher" if v > 0 else "the document shows no positive amount"
                sev, conf = ("high", 0.85) if has_gross else ("medium", 0.7)
                extra = "" if has_gross else (" No payslip gross is available, so the declared figure may be gross while the documents show "
                                              "net pay or credits; verify which basis was declared.")
                out.append(Finding(nid(), "cross", f"Declared income exceeds income shown in {d.doc_id}", sev, conf, "inconsistency",
                                   f"Declared monthly income {decl:,.0f} vs the highest amount in the documents, {d.doc_id} "
                                   f"{k.replace('_', ' ')} {v:,.0f} ({pct})." + extra,
                                   [_ref(fd, "declared_income"), _ref(d, k)], method="income reconciliation"))
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
