"""Contextual verification (5.5) and cross-document correlation (5.6).
Every finding carries refs back to (doc, field, bbox) so it is traceable to the source."""
import os, re
from datetime import date
from rapidfuzz import fuzz
from .schema import Finding
from .compare import compare
from .extract import KIND_NAMES, DUP_STRICT, is_misplaced, EXPECTED

REQUIRED = {
    "application_form": ["name", "dob", "id_number", "declared_income"],
    "id_card": [],
    "salary_slip": ["name", "gross", "deductions", "net_pay"],
    "bank_statement": ["name", "account_no"],
    "employment_letter": ["name", "employer"],
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


def _misplaced_finding(nid, d, u):
    key = u["key"]
    sev = "medium" if key in ("dob", "id_number", "account_no", "name") else "low"
    return Finding(nid(), "intra", f"Value under '{key}' looks like {KIND_NAMES.get(u['kind'], u['kind'])}, not a {key.replace('_', ' ')} ({d.doc_id})",
                   sev, 0.6, "observed",
                   f"The label '{key.replace('_', ' ')}' was found, but the value next to it ({u['raw']!r}) looks like {KIND_NAMES.get(u['kind'], u['kind'])}. "
                   "The value may be misplaced (typed or pasted under the wrong label), the label may be misread, or the layout is unusual. "
                   "The field was left empty rather than filled with a value of the wrong kind. Verify manually.",
                   [{"doc_id": d.doc_id, "field": key, "value": u["raw"], "bbox": u.get("bbox")}], method="value-kind check")


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
        exp_keys = EXPECTED.get(d.doc_type, set())               # a label that does not belong to this kind of document is not a missing field
        seen_u = set()
        for u in d.facts.get("unparsed", []):
            if u["key"] in seen_u or d.get(u["key"]) or u["key"] not in exp_keys:          # one report per field; skip if another pass read it
                continue
            seen_u.add(u["key"])
            if u.get("kind") and is_misplaced(u["key"], u["kind"]):       # readable, but it is the wrong kind of value for this label
                out.append(_misplaced_finding(nid, d, u))
                continue
            out.append(Finding(nid(), "missing", f"Value of '{u['key']}' could not be read in {d.doc_id}", "low", 0.7, "observed",
                               f"The label was found but its value ({u['raw']!r}) is not a valid {u['key'].replace('_', ' ')}. Checks that need it "
                               f"(arithmetic, cross-document comparison) were skipped for this field. Verify manually.",
                               [{"doc_id": d.doc_id, "field": u["key"], "value": u["raw"], "bbox": None}], method="value validation"))
        seen_r = set()
        for u in d.facts.get("rejected", []):                              # values dropped by the type guards because they look like something else
            if u["key"] in seen_r or d.get(u["key"]) or u["key"] not in exp_keys:
                continue
            seen_r.add(u["key"])
            out.append(_misplaced_finding(nid, d, u))
        for dp in d.facts.get("duplicates", [])[:6]:                        # same single-valued field twice with different values
            strict = dp["key"] in DUP_STRICT
            label = _LABELS.get(dp["key"], dp["key"].replace("_", " "))
            out.append(Finding(nid(), "intra", f"Two different values for '{dp['key']}' on {d.doc_id}", "medium" if strict else "low",
                               0.75 if strict else 0.55, "inconsistency" if strict else "observed",
                               f"{label} appears twice on the same page with different values: {dp['kept']['raw']!r} (used) and {dp['other']['raw']!r}. "
                               + ("A single-valued field should not differ within one page; one of them may have been overwritten or altered. "
                                  if strict else "This may be a second party or a different column (for example year-to-date), or an OCR/overlay problem. ")
                               + "Verify against the original.",
                               [{"doc_id": d.doc_id, "field": dp["key"], "value": dp["kept"]["raw"], "bbox": dp["kept"]["bbox"]},
                                {"doc_id": d.doc_id, "field": dp["key"], "value": dp["other"]["raw"], "bbox": dp["other"]["bbox"]}],
                               method="duplicate field check"))
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
        # annual gross = 12 x monthly gross
        mo = d.get("monthly_gross") or d.get("declared_income")
        an = d.get("annual_gross")
        if mo and an and isinstance(mo.value, float) and isinstance(an.value, float) and mo.value > 0:
            if abs(an.value - 12 * mo.value) > max(1.0, 0.01 * an.value):
                out.append(Finding(nid(), "intra", f"Annual income is not 12 times the monthly income in {d.doc_id}", "medium", 0.8, "inconsistency",
                                   f"Monthly gross {mo.value:,.0f} x 12 = {12 * mo.value:,.0f}, but the annual figure is {an.value:,.0f} "
                                   f"(difference {an.value - 12 * mo.value:+,.0f}). An annual figure that includes bonuses would explain a higher value; verify.",
                                   [_ref(d, mo.key), _ref(d, "annual_gross")], method="annual vs monthly income"))
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
           "epic_no": "Voter ID (EPIC) number", "father_name": "Father's name", "employee_id": "Employee ID"}
_EXACT_KEYS = {"dob", "id_number", "account_no", "aadhaar_no", "epic_no", "employee_id"}


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
            sev = "high" if key in ("name", "dob", "id_number", "account_no") else "medium"
            if r.verdict == "near" and not exact_kind:
                sev = "medium"                         # a text near-miss is more often an OCR slip than an alteration
            weak = min(_min_conf(og, key), _min_conf(rg, key)) < 0.6
            if weak:
                sev = _DOWN[sev]
            exact_text = all(d.born_digital and d.pdf for d in og + rg)              # read from the PDF text layer: not an OCR slip
            note = (("Values differ only slightly (similarity %d%%): possible typo, transliteration or minor alteration (the text was read exactly from the PDF, so this is not an OCR slip)." if exact_text
                     else "Values differ only slightly (similarity %d%%): possible OCR slip, typo, transliteration or minor alteration.") % r.sim
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


def _payroll_credits(docs):
    """Payroll-like credits of every bank statement page: dicts with amount, date, box, the document and a ref."""
    out = []
    for d in docs:
        if d.doc_type != "bank_statement":
            continue
        cs = d.facts.get("payroll_credits") or []
        for c in cs:
            out.append({"doc": d, "amount": c["amount"], "date": c["date"], "desc": c["desc"], "payer": c.get("payer"),
                        "ref": {"doc_id": d.doc_id, "field": "payroll_credit", "value": c.get("raw") or f"{c['amount']:,.0f}", "bbox": list(c["bbox"])}})
        if not cs and _num(d, "salary_credit") is not None:
            out.append({"doc": d, "amount": _num(d, "salary_credit"), "date": None, "desc": "", "payer": None, "ref": _ref(d, "salary_credit")})
    return out


def _iso_gap(a, b):
    try:
        return abs((date.fromisoformat(a) - date.fromisoformat(b)).days)
    except (TypeError, ValueError):
        return None


def _amount_checks(docs, nid, skip_declared=False):
    """Monthly gross income stated in the application, the employer's confirmation and the payslip must agree; so must the annual figure.
    skip_declared: the declared figure was already reported as exceeding the documents, so it is not reported a second time here."""
    out = []
    srcs = []                                                       # (doc, field key, value, label)
    for d in docs:
        for key, label in (("declared_income", "declared monthly income"), ("monthly_gross", "monthly gross salary"), ("gross", "payslip gross pay")):
            if key == "declared_income" and (skip_declared or "gross" not in (d.get(key).label.lower() if d.get(key) else "")):
                continue                                              # only a figure the form itself calls GROSS is compared with gross pay
            v = _num(d, key)
            if v is not None and d.doc_type in ("application_form", "employment_letter", "salary_slip"):
                srcs.append((d, key, v, label))
    if len({round(v) for _, _, v, _ in srcs}) > 1 and len(srcs) >= 2:
        hi, lo = max(srcs, key=lambda x: x[2]), min(srcs, key=lambda x: x[2])
        if hi[2] - lo[2] > max(1.0, 0.01 * hi[2]):
            out.append(Finding(nid(), "cross", "Monthly gross income differs between documents", "high", 0.85, "inconsistency",
                               "; ".join(f"{d.doc_id}: {label} {v:,.0f}" for d, _, v, label in srcs)
                               + f". The highest ({hi[0].doc_id}) is {(hi[2] / lo[2] - 1) * 100:.1f}% above the lowest ({lo[0].doc_id}).",
                               [_ref(d, k) for d, k, _, _ in srcs], method="gross income comparison"))
    ann = [(d, _num(d, "annual_gross")) for d in docs if _num(d, "annual_gross") is not None]
    if len({round(v) for _, v in ann}) > 1:
        hi, lo = max(ann, key=lambda x: x[1]), min(ann, key=lambda x: x[1])
        out.append(Finding(nid(), "cross", "Annual gross income differs between documents", "high", 0.85, "inconsistency",
                           "; ".join(f"{d.doc_id}: {v:,.0f}" for d, v in ann) + ".", [_ref(d, "annual_gross") for d, _ in ann], method="annual income comparison"))
    return out


_MON3 = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def _month_key(raw):
    """'October 2026' / 'Oct-2026' -> '2026-10'; None when no month and year can be read."""
    m = re.search(r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", str(raw or ""))
    y = re.search(r"\b(19|20)\d{2}\b", str(raw or ""))
    return f"{y.group(0)}-{_MON3.index(m.group(1).lower()) + 1:02d}" if m and y else None


def _same_month_slips(docs, nid):
    """Two payslips of the same employee for the same month must show the same pay. A copy with one amount changed is exactly this."""
    out, groups = [], {}
    for d in docs:
        if d.doc_type != "salary_slip":
            continue
        month = _month_key(d.get("pay_month").raw if d.get("pay_month") else None)
        who = (d.get("employee_id").value if d.get("employee_id") else None) or (d.get("id_number").value if d.get("id_number") else None) \
            or (d.get("name").value if d.get("name") else None)
        if month and who:
            groups.setdefault((str(who).upper(), month), []).append(d)
    for (who, month), ds in groups.items():
        if len(ds) < 2:
            continue
        for key, label in (("net_pay", "net pay"), ("gross", "gross pay"), ("deductions", "deductions")):
            have = [(d, _num(d, key)) for d in ds if _num(d, key) is not None]
            if len(have) < 2:
                continue
            hi, lo = max(have, key=lambda x: x[1]), min(have, key=lambda x: x[1])
            if hi[1] - lo[1] > max(1.0, 0.01 * hi[1]):
                out.append(Finding(nid(), "cross", f"Payslips for the same employee and month show different {label}", "high", 0.9, "inconsistency",
                                   "; ".join(f"{d.doc_id}: {v:,.0f}" for d, v in have) + f" (employee {who}, {month}). Two payslips for one month should agree; "
                                   f"either one was changed or they come from different sources. This is a data conflict to verify with the employer, not proof of fraud.",
                                   [_ref(d, key) for d, _ in have], metrics={key: [v for _, v in have], "difference": round(hi[1] - lo[1], 2)},
                                   method="same-month payslip comparison"))
    return out


def _income_checks(docs, nid):
    """Declared income vs payslips vs bank credits, across ALL payslips / statement pages (not one per type).
    People often declare GROSS monthly income while the bank shows NET credit, so a declared amount is only flagged
    HIGH when it exceeds even the largest amount any document shows; against net-only evidence it is flagged as a
    medium 'verify' item that names the gross/net possibility."""
    out = _same_month_slips(docs, nid)
    slips = [(d, _num(d, "net_pay"), _num(d, "gross")) for d in docs if d.doc_type == "salary_slip" and _num(d, "net_pay") is not None]
    credits = _payroll_credits(docs)
    forms = [d for d in docs if d.doc_type == "application_form" and _num(d, "declared_income") is not None]
    close = lambda a, b: abs(a - b) <= 0.01 * max(abs(a), abs(b), 1.0)
    for sd, net, _gross in slips:                                   # each payslip must be matched by some bank credit
        if not credits or any(close(net, c["amount"]) for c in credits):
            continue
        pay = sd.get("pay_date")
        pd_iso = pay.value if pay and isinstance(pay.value, str) else None

        def rank(c):
            gap = _iso_gap(pd_iso, c["date"])
            return (0 if gap is not None else 1, gap if gap is not None else 0, abs(c["amount"] - net))
        c = min(credits, key=rank)
        when = f" on {c['date']}" if c["date"] else ""
        pay_txt = f" (payment date {pay.raw})" if pay else ""
        out.append(Finding(nid(), "cross", "Salary slip net pay does not match bank payroll credit", "high", 0.9, "inconsistency",
                           f"Net pay on {sd.doc_id} is {net:,.0f}{pay_txt}, but the payroll credit in the bank statement ({c['doc'].doc_id}){when} is {c['amount']:,.0f} "
                           f"(difference {net - c['amount']:+,.0f}). Either the payslip or the bank record is wrong; this is a data inconsistency to verify with the "
                           f"employer and the bank, not proof of fraud.",
                           [_ref(sd, "net_pay"), c["ref"]], metrics={"net_pay": net, "bank_credit": c["amount"], "difference": round(net - c["amount"], 2)},
                           method="income reconciliation"))
    for sd, net, _g in slips:                                       # payslip employer vs payer named in the bank narration
        emp = sd.get("employer")
        for c in credits:
            if emp and c["payer"] and close(net, c["amount"]):
                r = compare("employer", emp.value, c["payer"])
                if r.verdict in ("near", "different"):
                    out.append(Finding(nid(), "cross", f"Payroll payer in the bank statement differs from the payslip employer", "medium", 0.7, "inconsistency",
                                       f"{sd.doc_id} names {emp.raw!r}; the matching credit in {c['doc'].doc_id} is from {c['payer']!r}.",
                                       [_ref(sd, "employer"), c["ref"]], method="payer vs employer"))
    for fd in forms:
        decl = _num(fd, "declared_income")
        # A payslip the bank contradicts cannot vouch for a declared amount (a forger can raise both together), so when
        # bank credits exist only payslips whose net pay matches a credit count as supporting evidence.
        trusted = [t for t in slips if not credits or any(close(t[1], c["amount"]) for c in credits)]
        letters = [(ld, "monthly_gross", _num(ld, "monthly_gross")) for ld in docs
                   if ld.doc_type == "employment_letter" and _num(ld, "monthly_gross") is not None]        # the employer's own statement of gross pay
        amounts = [(sd, "gross", g) for sd, _n, g in trusted if g is not None] + [(sd, "net_pay", n) for sd, n, _g in trusted] + \
                  [(c["doc"], "salary_credit", c["amount"]) for c in credits] + letters
        if not amounts:
            continue
        has_gross = any(k == "gross" for _, k, _ in amounts)
        for d, k, v in sorted(amounts, key=lambda x: -x[2])[:1]:
            if decl > 1.05 * v:
                pct = f"{(decl / v - 1) * 100:.0f}% higher" if v > 0 else "the document shows no positive amount"
                sev, conf = ("high", 0.85) if has_gross else ("medium", 0.7)
                extra = "" if has_gross else (" No payslip gross is available, so the declared figure may be gross while the documents show "
                                              "net pay or credits; verify which basis was declared.")
                ref_other = _ref(d, k) if d.get(k) else next((c["ref"] for c in credits if c["doc"] is d and close(c["amount"], v)), _ref(d, k) if d.get(k) else None)
                out.append(Finding(nid(), "cross", f"Declared income exceeds income shown in {d.doc_id}", sev, conf, "inconsistency",
                                   f"Declared monthly income {decl:,.0f} vs the highest amount in the documents, {d.doc_id} "
                                   f"{k.replace('_', ' ')} {v:,.0f} ({pct})." + extra,
                                   [_ref(fd, "declared_income")] + ([ref_other] if ref_other else []), method="income reconciliation"))
    out.extend(_amount_checks(docs, nid, skip_declared=any(f.title.startswith("Declared income exceeds") for f in out)))
    return out


def _overlap(a, b):
    """Share of the smaller box covered by the other (boxes as [x0, y0, x1, y1]); 0 when either is missing."""
    try:
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        if w <= 0 or h <= 0:
            return 0.0
        return w * h / max(1e-9, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))
    except (TypeError, IndexError):
        return 0.0


def corroborate(visual, others):
    """If a content inconsistency touches a field that also shows visual tampering evidence (same field, or its box lies in the
    visual finding's box), raise severity/confidence: two independent lines of evidence. Only visual findings that point to
    possible manipulation count, and whole-page references (no field, no box) never corroborate anything."""
    vis = [v for v in visual if v.status == "potential_manipulation"]
    for o in others:
        hit = []
        for v in vis:
            for ro in o.refs:
                if ro.get("field") is None and not ro.get("bbox"):
                    continue
                for rv in v.refs:
                    if rv["doc_id"] != ro["doc_id"]:
                        continue
                    same_field = ro.get("field") and ro.get("field") == rv.get("field")
                    in_box = rv.get("bbox") and ro.get("bbox") and _overlap(ro["bbox"], rv["bbox"]) >= 0.6
                    if same_field or in_box:
                        hit.append(v)
                        break
                else:
                    continue
                break
        hit = list({v.fid: v for v in hit}.values())
        if hit:
            o.severity = "high"
            o.confidence = round(min(0.98, o.confidence + 0.05), 2)
            o.explanation += f" Corroborated by visual evidence ({', '.join(h.fid for h in hit)}) on the same field."
            o.metrics["corroborated_by"] = [h.fid for h in hit]
            for h in hit:
                h.confidence = round(min(0.9, h.confidence + (0.02 if h.metrics.get("corroborated_by") else 0.10)), 2)   # two independent lines of evidence; the first one counts most
                h.metrics.setdefault("corroborated_by", []).append(o.fid)
    return others
