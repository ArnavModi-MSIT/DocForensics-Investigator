"""Contextual verification, part 2 (5.5): rules that relate fields to each other, to dates and to table contents.

Each rule is conservative (a rule that cannot be evaluated stays silent), states exactly what was compared, and is an indicator:
  PAN vs holder name     4th PAN letter = holder type (P = individual), 5th = initial of the surname
  Date logic             no future dates; statement period start <= end; pay month not in the future
  Age                    an 'Age: N' printed next to the date of birth must match the DOB (as of today or of the 'as on' date)
  Statement consistency  transactions inside the statement period; header opening balance = first row; header closing balance = last
                         row; a 'salary credit' header amount must appear among the deposits
  Payslip vs bank        a payslip's net pay should appear as a deposit in the month it was earned or the next, and the deposit's
                         narration should not name a different company
"""
import os, re
from datetime import date, timedelta
from rapidfuzz import fuzz
from .schema import Finding
from .compare import name_tokens, employer_tokens
from .extract import DATE_PATTERNS, parse_date, MONTH_RE, YEAR_RE, MONTH_NUM

PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
PAN_HOLDER = {"P": "an individual", "C": "a company", "H": "a Hindu Undivided Family", "F": "a firm", "A": "an association of persons",
              "T": "a trust", "B": "a body of individuals", "L": "a local authority", "J": "an artificial juridical person", "G": "a government body"}
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
GENERIC = {"salary", "sal", "credit", "neft", "imps", "upi", "rtgs", "ach", "payroll", "wages", "transfer", "from", "ref", "txn", "payment", "cr", "inward",
           "remittance", "for", "the", "month", "pay", "bank", "ltd", "limited", "pvt", "private", "and"}


def dates_in(text, ordered=False):
    """ISO dates found in the text; sorted by value, or in the order they are printed when ordered=True."""
    found = {}
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, text, re.I):
            iso = parse_date(m.group(0))
            if iso and iso not in found:
                found[iso] = m.start()
    return [k for k, _ in sorted(found.items(), key=lambda kv: kv[1])] if ordered else sorted(found)


def parse_month(raw):
    """(year, month) from 'October 2026', 'Oct-26', '10/2026'; None when no month can be read."""
    t = str(raw)
    m = MONTH_RE.search(t)
    if m:
        mon = MONTHS[m.group(1).lower()[:3]]
        y = YEAR_RE.search(t)
        if y:
            return int(y.group(0)), mon
        y2 = re.search(r"(?<!\d)(\d{2})(?!\d)", t[m.end():])
        return (2000 + int(y2.group(1)), mon) if y2 else None
    n = MONTH_NUM.search(t)
    if n:
        a, b = re.split(r"[/\-]", n.group(0))
        return (int(b), int(a)) if 1 <= int(a) <= 12 else None
    return None


def parse_period(raw, ordered=False):
    """(start_iso or None, end_iso or None) from a statement period / statement date. ordered=True keeps the printed order
    (so a period that ends before it starts can be seen); otherwise the dates are sorted."""
    ds = dates_in(str(raw), ordered)
    if len(ds) >= 2:
        return ds[0], ds[-1]
    if len(ds) == 1:
        return None, ds[0]
    ym = parse_month(raw)
    if ym:
        y, m = ym
        end = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
        return date(y, m, 1).isoformat(), end.isoformat()
    return None, None


def _ref(doc, key):
    f = doc.get(key)
    return {"doc_id": doc.doc_id, "field": key, "value": f.raw, "bbox": list(f.bbox)}


def _row_ref(doc, t, r, label):
    bb = r["cells"].get("date", {}).get("bbox") or r["bbox"]
    return {"doc_id": doc.doc_id, "field": f"{t['tid']} row {r['n']} {label}", "value": r.get("date"), "bbox": list(bb)}


def _num(d, key):
    f = d.get(key)
    return f.value if f and isinstance(f.value, float) else None


def _tables(ds):
    return [(d, t) for d in ds for t in d.tables if t["kind"] == "transactions"]


class _Ids:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return f"K-{self.n:03d}"


# ------------------------------------------------------------------------------------------- per-document rules
def _placeholder_pan(pan):
    """Specimen / dummy PANs ('ABCDE1234F', 'AAAAA0000A', 'XXXXX9999X') carry no meaning in their letters, so the structure rules skip them."""
    letters, digits = pan[:5], pan[5:9]
    seq = all(ord(b) - ord(a) == 1 for a, b in zip(letters, letters[1:]))
    return seq or len(set(letters)) == 1 or digits in ("1234", "0000", "1111", "9999", "4321", "5678", "0123")


def _pan_vs_name(d, out, nid):
    f, nm = d.get("id_number"), d.get("name")
    if d.doc_type not in ("application_form", "id_card") or not f or not nm or not PAN_RE.match(str(f.value)):
        return
    toks = name_tokens(nm.value)
    if len(toks) < 2 or _placeholder_pan(str(f.value)):
        return
    pan = str(f.value)
    if pan[3] != "P":
        out.append(Finding(nid(), "intra", f"PAN holder type does not fit an individual in {d.doc_id}", "medium", 0.55, "inconsistency",
                           f"The 4th character of PAN {pan} is '{pan[3]}', which denotes {PAN_HOLDER.get(pan[3], 'a non-individual')}, but the document gives "
                           f"a person's name ({nm.raw!r}). For an individual the 4th character is 'P'.",
                           [_ref(d, "id_number"), _ref(d, "name")], method="PAN structure check"))
    elif pan[4] not in (toks[-1][0].upper(), toks[0][0].upper()):
        out.append(Finding(nid(), "intra", f"PAN does not match the holder's surname initial in {d.doc_id}", "low", 0.5, "observed",
                           f"The 5th character of an individual's PAN is the initial of the surname. PAN {pan} has '{pan[4]}', but the name {nm.raw!r} "
                           f"gives '{toks[-1][0].upper()}' (last word) or '{toks[0][0].upper()}' (first word). A changed surname can explain this; otherwise verify.",
                           [_ref(d, "id_number"), _ref(d, "name")], method="PAN structure check"))


def _future_dates(d, out, nid, today):
    lim = (today + timedelta(days=1)).isoformat()
    for key, sev, conf in (("dob", "high", 0.9), ("issue_date", "high", 0.85)):
        f = d.get(key)
        if f and isinstance(f.value, str) and re.match(r"\d{4}-\d{2}-\d{2}$", f.value) and f.value > lim:
            out.append(Finding(nid(), "intra", f"{'Date of birth' if key == 'dob' else 'Issue date'} lies in the future in {d.doc_id}", sev, conf, "inconsistency",
                               f"{f.raw!r} ({f.value}) is later than today ({today.isoformat()}).", [_ref(d, key)], method="date logic"))
    p = d.get("period")
    if p:
        s, e = parse_period(p.raw, ordered=True)
        if s and e and s > e:
            out.append(Finding(nid(), "intra", f"Statement period ends before it starts in {d.doc_id}", "high", 0.85, "inconsistency",
                               f"The period {p.raw!r} runs from {s} to {e}.", [_ref(d, "period")], method="date logic"))
        elif e and e > (today + timedelta(days=2)).isoformat():
            out.append(Finding(nid(), "intra", f"Statement period/date lies in the future in {d.doc_id}", "medium", 0.75, "inconsistency",
                               f"{p.raw!r} ends on {e}, after today ({today.isoformat()}).", [_ref(d, "period")], method="date logic"))
    pm = d.get("pay_month")
    if pm:
        ym = parse_month(pm.raw)
        if ym and ym > (today.year, today.month):
            out.append(Finding(nid(), "intra", f"Pay month lies in the future in {d.doc_id}", "medium", 0.7, "inconsistency",
                               f"{pm.raw!r} is later than the current month.", [_ref(d, "pay_month")], method="date logic"))


AGE_RE = re.compile(r"(?i)\bage\b(?:\s*as\s*on\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}))?[^0-9A-Za-z]{0,6}(\d{1,3})\b(?![./-]\d)")


def _age_vs_dob(d, out, nid, today):
    dob = d.get("dob")
    if d.doc_type not in ("application_form", "id_card") or not dob or not re.match(r"\d{4}-\d{2}-\d{2}$", str(dob.value)):
        return
    text = " ".join(w.text for w in d.words)
    m = AGE_RE.search(text)
    if not m or not 1 <= int(m.group(2)) <= 120:
        return
    ref = today
    if m.group(1):
        iso = parse_date(m.group(1).replace(" ", ""))
        ref = date.fromisoformat(iso) if iso else today
    y, mo, dd = map(int, dob.value.split("-"))
    implied = ref.year - y - ((ref.month, ref.day) < (mo, dd))
    stated = int(m.group(2))
    if abs(implied - stated) > 1:
        out.append(Finding(nid(), "intra", f"Stated age does not match the date of birth in {d.doc_id}", "medium", 0.65, "inconsistency",
                           f"The page prints an age of {stated}, but the date of birth {dob.raw} implies {implied} "
                           f"{'on ' + ref.isoformat() if m.group(1) else 'today'}.", [_ref(d, "dob")], method="age consistency"))


# ------------------------------------------------------------------------------------------- bank statement consistency
def _statement(ds, out, nid, path):
    txs = _tables(ds)
    if not txs:
        return
    name = os.path.basename(path)
    # transactions inside the statement period
    pf = next((d.get("period") for d in ds if d.get("period")), None)
    if pf:
        s, e = parse_period(pf.raw)
        bad = []
        for d, t in txs:
            for r in t["rows"]:
                if r["role"] != "txn" or not r["date"]:
                    continue
                if (s and r["date"] < s) or (e and r["date"] > e):
                    bad.append((d, t, r))
        if bad:
            lo = f"from {s} " if s else ""
            out.append(Finding(nid(), "intra", f"{len(bad)} transaction(s) lie outside the statement period ({name})", "medium", 0.7, "inconsistency",
                               f"The statement period/date reads {pf.raw!r} (valid {lo}to {e}), but " + "; ".join(
                                   f"{d.doc_id} row {r['n']} is dated {r['date']}" for d, t, r in bad[:4]) + (f" (+{len(bad) - 4} more)" if len(bad) > 4 else "") + ".",
                               [_row_ref(d, t, r, "date") for d, t, r in bad[:6]], method="statement period check"))
    # header opening balance = first row; closing balance = last row
    first_d, first_t = txs[0]
    rows = first_t["rows"]
    ob = next(((d, _num(d, "opening_balance")) for d in ds if _num(d, "opening_balance") is not None), None)
    if ob and rows:
        r0 = rows[0]
        derived = r0["balance"] if r0["role"] == "opening" else (None if r0["balance"] is None else r0["balance"] - (r0["credit"] or 0) + (r0["debit"] or 0))
        if derived is not None and abs(derived - ob[1]) > 0.01:
            out.append(Finding(nid(), "intra", f"Opening balance in the header differs from the first table row ({name})", "high", 0.75, "inconsistency",
                               f"The header gives an opening balance of {ob[1]:,.2f}, but the first table row implies {derived:,.2f}.",
                               [_ref(ob[0], "opening_balance"), _row_ref(first_d, first_t, r0, "balance")], method="statement balance check"))
    cb = next(((d, _num(d, "closing_balance")) for d in ds if _num(d, "closing_balance") is not None), None)
    last = next(((d, t, r) for d, t in reversed(txs) for r in reversed(t["rows"]) if r["role"] == "txn" and r["balance"] is not None), None)
    if cb and last and abs(last[2]["balance"] - cb[1]) > 0.01:
        out.append(Finding(nid(), "intra", f"Closing balance in the header differs from the last table row ({name})", "high", 0.75, "inconsistency",
                           f"The header gives a closing balance of {cb[1]:,.2f}, but the last transaction row ends at {last[2]['balance']:,.2f}.",
                           [_ref(cb[0], "closing_balance"), _row_ref(last[0], last[1], last[2], "balance")], method="statement balance check"))
    # a salary credit stated in the header must be among the deposits
    sc = next(((d, _num(d, "salary_credit")) for d in ds if _num(d, "salary_credit") is not None), None)
    credits = [r["credit"] for d, t in txs for r in t["rows"] if r["role"] == "txn" and r["credit"]]
    if sc and credits and not any(abs(c - sc[1]) <= 0.01 for c in credits):
        out.append(Finding(nid(), "intra", f"Salary credit in the header is not among the deposits ({name})", "medium", 0.6, "inconsistency",
                           f"The header states a salary credit of {sc[1]:,.2f}, but no deposit row has that amount.", [_ref(sc[0], "salary_credit")],
                           method="statement credit check"))


# ------------------------------------------------------------------------------------------- payslip vs bank statement
def _next_month(y, m):
    return (y + (m == 12), m % 12 + 1)


def _payslip_vs_bank(docs, out, nid):
    slips = [d for d in docs if d.doc_type == "salary_slip" and _num(d, "net_pay") and d.get("pay_month") and parse_month(d.get("pay_month").raw)]
    rows = [(d, t, r) for d in docs if d.doc_type == "bank_statement" for t in d.tables if t["kind"] == "transactions"
            for r in t["rows"] if r["role"] == "txn" and r["credit"] and r["date"]]
    if not slips or not rows:
        return
    for sd in slips:
        net, ym = _num(sd, "net_pay"), parse_month(sd.get("pay_month").raw)
        hit = [x for x in rows if abs(x[2]["credit"] - net) <= max(1.0, 0.005 * net)]
        if not hit:
            continue                                                            # amount mismatch is reported by the income reconciliation
        hit.sort(key=lambda x: 0 if re.search(r"sal", x[2]["desc"], re.I) else 1)
        bd, bt, br = hit[0]
        cy, cm = int(br["date"][:4]), int(br["date"][5:7])
        if (cy, cm) not in (ym, _next_month(*ym)):
            out.append(Finding(nid(), "cross", f"Salary credit is not in the pay month or the month after ({sd.doc_id})", "low", 0.5, "observed",
                               f"The payslip is for {sd.get('pay_month').raw!r} with net pay {net:,.2f}; the matching deposit in {bd.doc_id} is dated {br['date']} "
                               "(salary is normally credited in the pay month or the next). It may be an advance or a late payment.",
                               [_ref(sd, "pay_month"), _row_ref(bd, bt, br, "credit")], method="payslip vs bank month check"))
        emp = sd.get("employer")
        if emp:
            etoks = [w for w in employer_tokens(emp.value) if len(w) >= 4]
            ntoks = [w for w in re.findall(r"[a-z]{3,}", br["desc"].lower()) if w not in GENERIC]
            if etoks and ntoks and not any(fuzz.ratio(a, b) >= 85 for a in etoks for b in ntoks):
                out.append(Finding(nid(), "cross", f"Salary credit narration does not mention the payslip employer ({sd.doc_id})", "low", 0.5, "observed",
                                   f"The payslip employer is {emp.raw!r}, but the narration of the matching deposit ({br['desc'][:50]!r}) names something else. "
                                   "Narrations are often abbreviated, so verify who paid the salary.",
                                   [_ref(sd, "employer"), _row_ref(bd, bt, br, "credit")], method="payslip vs bank employer check"))


# ------------------------------------------------------------------------------------------- driver
def consistency_findings(docs, today=None):
    today = today or date.today()
    out, nid = [], _Ids()
    for d in docs:
        if d.doc_type == "unknown":
            continue
        _pan_vs_name(d, out, nid)
        _future_dates(d, out, nid, today)
        _age_vs_dob(d, out, nid, today)
    by_file = {}
    for d in docs:
        if d.doc_type == "bank_statement":
            by_file.setdefault(d.path, []).append(d)
    for path, ds in by_file.items():
        _statement(ds, out, nid, path)
    _payslip_vs_bank(docs, out, nid)
    return out
