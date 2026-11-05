import os, re
from datetime import datetime
from rapidfuzz import fuzz
from .schema import Field
from .ocr import group_lines

# ---- document type classification (keyword scoring over OCR text) ----------
DOC_TYPES = {
    "application_form": ["application", "applicant", "declared", "loan"],
    "id_card": ["national", "identity", "card", "issue"],
    "salary_slip": ["salary", "slip", "payslip", "earnings", "net pay", "deductions"],
    "bank_statement": ["bank", "statement", "account", "balance", "credits"],
    "employment_letter": ["employment", "employee", "employer", "confirmation", "income", "designation", "joining", "salary"],
}

# ---- label dictionary per field key (fuzzy matched) -------------------------
LABELS = {
    "name": ["applicant name", "applicant full name", "name", "account holder", "employee name", "employee full name", "declared name",
             "holder name", "electors name", "elector name"],
    "dob": ["date of birth", "date of birth / age", "date of birth age", "date of birth/age", "dob", "birth date", "declared date of birth"],
    "father_name": ["fathers name", "father name", "father's name", "husbands name", "husband name"],
    "id_number": ["id number", "pan", "id no", "pan number", "synthetic identity reference", "synthetic identifier", "identity reference",
                  "identifier", "personal id", "personal identifier"],
    "address": ["address", "residential address", "residential address on file", "current residential address", "address on file"],
    "employer": ["employer", "company", "organisation", "employer name"],
    "employee_id": ["employee id", "employee code", "employee no", "employee number", "emp id", "staff id"],
    "declared_income": ["declared monthly income", "monthly income", "declared income"],
    "monthly_gross": ["monthly gross salary", "monthly gross income", "gross monthly salary", "gross monthly income"],
    "annual_gross": ["annual gross income", "annualized gross salary", "annualised gross salary", "annual gross salary", "annual income",
                     "annual salary", "yearly gross income"],
    "credit_amount": ["credit amount requested", "amount requested", "loan amount requested", "requested loan amount", "loan amount"],
    "term": ["requested repayment term", "repayment term", "loan tenure", "tenure"],
    "employment_type": ["employment type", "type of employment", "employment status"],
    "start_date": ["employment start date", "date of joining", "joining date", "start date"],
    "pay_date": ["payment date", "pay date", "date of payment", "salary payment date"],
    "review_date": ["review due date", "review date"],
    "case_ref": ["case reference", "case no", "case number", "application reference"],
    "doc_date": ["submitted", "signed", "generated", "date submitted", "submitted on", "signed on", "generated on"],
    "issue_date": ["issue date", "date of issue", "issued date"],
    "gross": ["gross", "gross earnings", "gross salary", "gross pay", "gross amount", "total gross", "gross total"],
    "deductions": ["total deductions", "deductions"],
    "net_pay": ["net pay", "net salary", "take home", "net pay / take home"],
    "pay_month": ["pay month", "month", "salary month", "payroll period", "period confirmed", "salary period"],
    "account_no": ["account number", "account no", "a/c no", "salary account number", "salary account no", "salary account", "bank a/c", "bank account",
                   "bank account number", "bank account no"],
    "period": ["statement period", "statement date", "period"],
    "opening_balance": ["opening balance", "balance forward", "balance brought forward", "balance b/f"],
    "salary_credit": ["salary credit", "salary credited"],
    "closing_balance": ["closing balance"],
    "currency": ["currency"],
    "branch": ["branch"],
    "ifsc": ["ifsc", "ifsc code"],
    "account_type": ["account type", "account class"],
}
DATE_KEYS = {"dob", "issue_date", "start_date", "pay_date", "review_date", "doc_date"}
MONEY_KEYS = {"declared_income", "gross", "deductions", "net_pay", "monthly_gross", "annual_gross", "credit_amount",
              "opening_balance", "salary_credit", "closing_balance"}


EXPECTED = {   # fields that genuinely belong to each type (used as structural evidence)
    "application_form": {"name", "dob", "id_number", "address", "employer", "declared_income", "monthly_gross", "annual_gross",
                         "employee_id", "credit_amount"},
    "id_card": {"name", "father_name", "dob", "id_number", "aadhaar_no", "epic_no", "address", "issue_date", "case_ref", "review_date"},
    "salary_slip": {"name", "employer", "pay_month", "gross", "deductions", "net_pay", "employee_id", "pay_date"},
    "bank_statement": {"name", "account_no", "period", "opening_balance", "salary_credit", "closing_balance",
                       "currency", "branch", "ifsc", "account_type"},
    "employment_letter": {"name", "employee_id", "employer", "employment_type", "start_date", "monthly_gross", "annual_gross", "pay_month"},
}
MIN_FIELDS = int(os.environ.get("DOCFOR_MIN_FIELDS", 3))          # a page is a supported document only if >= this many of its expected fields were found
STRONG_KW = float(os.environ.get("DOCFOR_STRONG_KW", 0.75))   # keyword score that counts as strong evidence
KW_CUTOFF = 0.5         # keyword evidence needed when no embedding model is available
KW_WEIGHT = float(os.environ.get("DOCFOR_KW_WEIGHT", 0.2))         # weight of keywords when mixing with embedding similarity (tunable)
VETO_MAX_FIELDS = int(os.environ.get("DOCFOR_VETO_MAX_FIELDS", 5))    # a page with this many labelled fields of a type is a form, not prose that merely resembles another class
NEG_MARGIN = float(os.environ.get("DOCFOR_NEG_MARGIN", 0.03))       # an "other:*" class must beat the best supported class by this cosine margin to veto


def keyword_hits(text, kws):
    """Whole-word / whole-phrase matching. Hyphens count as word characters, so
    'applicant-facing' does not match 'applicant' and 'issues' does not match 'issue'."""
    return [k for k in kws if re.search(r"(?<![a-z0-9-])" + re.escape(k) + r"(?![a-z0-9-])", text)]


SEM_LABEL_MIN = float(os.environ.get("DOCFOR_SEM_LABEL_MIN", 0.70))     # cosine needed for a semantic label match
SEM_LABEL_MARGIN = float(os.environ.get("DOCFOR_SEM_LABEL_MARGIN", 0.06))  # lead over the 2nd-best field key
EXTRA_LABELS = {   # synonyms used only by the semantic matcher (the fuzzy matcher keeps LABELS unchanged)
    "name": ["full name", "customer name", "name of employee", "account holder name", "name of applicant"],
    "dob": ["born on", "birth date"],
    "id_number": ["permanent account number", "identity number", "national id number", "document number"],
    "address": ["permanent address", "correspondence address", "current address"],
    "employer": ["name of employer", "employer name", "company name"],
    "declared_income": ["gross monthly income", "net monthly income", "monthly salary declared"],
    "issue_date": ["issued on"],
    "gross": ["total earnings", "monthly remuneration", "total salary", "gross total"],
    "deductions": ["total deduction", "less deductions"],
    "net_pay": ["net payable", "take home pay", "amount payable to employee"],
    "pay_month": ["pay period", "for the month of"],
    "account_no": ["bank account number", "acct no"],
    "period": ["statement for the period", "period covered"],
    "opening_balance": ["balance brought forward", "balance b/f", "starting balance"],
    "salary_credit": ["salary deposit", "credit salary", "monthly salary credit"],
    "closing_balance": ["balance carried forward", "ending balance", "available balance", "balance c/f"],
}
_label_cache = {"P": None, "keys": None}


def _label_protos(emb):
    if _label_cache["P"] is None:
        flat = [(k, p) for k in LABELS for p in LABELS[k] + EXTRA_LABELS.get(k, [])]
        _label_cache["keys"] = [k for k, _ in flat]
        _label_cache["P"] = emb.embed([p for _, p in flat])
    return _label_cache["keys"], _label_cache["P"]


MONTH_RE = re.compile(r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DATE_LIKE = re.compile(r"\d{1,4}[/\-.]\d{1,2}[/\-.]\d{2,4}")
MONTH_NUM = re.compile(r"\b\d{1,2}[/\-]\d{4}\b")


def _bad_label(text):
    """A label never contains digits and never has unbalanced brackets ('033 months)')."""
    return bool(re.search(r"\d", text)) or text.count("(") != text.count(")") or text.count("[") != text.count("]")


NOT_A_NAME = {"officer", "registration", "electoral", "constituency", "assembly", "parliamentary", "authority"}


def _value_ok(key, raw):
    """Type guard: the value must look like the field it is assigned to."""
    raw = raw.strip()
    if key in DATE_KEYS:
        return parse_date(raw) is not None
    if key in MONEY_KEYS:
        return parse_money(raw) is not None
    if key == "id_number":                           # report codes like RPT-2026-1507 have several hyphens: not an ID
        return (len(re.sub(r"\s", "", raw)) >= 6 and any(c.isdigit() for c in raw)
                and (raw.count("-") < 2 or bool(re.search(r"[A-Za-z]-\d{6,}$", raw))))      # DEMO-ID-713462 style references are fine
    if key == "employee_id":
        return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-/_.]{2,19}", raw)) and any(c.isdigit() for c in raw)
    if key == "case_ref":
        return len(raw) >= 5 and " " not in raw and any(c.isdigit() for c in raw)
    if key == "term":
        return bool(re.search(r"\d", raw)) and bool(re.search(r"(?i)month|year|yr|week|day", raw))
    if key == "employment_type":
        return sum(c.isalpha() for c in raw) >= 4 and "@" not in raw
    if key == "account_no":
        return sum(c.isdigit() for c in raw) >= 6 or _masked_number(raw) is not None
    if key == "period":                              # a date, a date range, or month + year
        return bool(DATE_LIKE.search(raw) or (MONTH_RE.search(raw) and YEAR_RE.search(raw)))
    if key == "pay_month":                           # month name, month/year or a year
        return bool(MONTH_RE.search(raw) or MONTH_NUM.search(raw) or (YEAR_RE.search(raw) and re.search(r"[A-Za-z]{3}", raw)))
    if key in ("name", "father_name"):               # person/company-like: few words, some capitalised, mostly letters
        toks = raw.split()
        t = re.sub(r"\s", "", raw)
        if any(re.sub(r"\W", "", w).lower() in NOT_A_NAME for w in toks):       # 'Registration Officer, 40 - NEW DELHI (GEN)'
            return False
        return (1 <= len(toks) <= 6 and len(t) >= 2 and "@" not in raw
                and sum(c.isdigit() for c in t) / len(t) < 0.2 and any(w[0].isupper() for w in toks))
    if key == "address":
        return len(raw) >= 8 and (any(c.isdigit() for c in raw) or "," in raw)
    if key == "employer":
        return sum(c.isalpha() for c in raw) >= 3 and "@" not in raw
    if key == "currency":
        return bool(re.fullmatch(r"[A-Za-z]{3}", raw.strip(" :,.")))
    if key == "ifsc":
        return bool(re.search(r"[A-Z]{4}0[A-Z0-9]{6}", raw.upper()))
    if key in ("branch", "account_type"):
        return sum(c.isalpha() for c in raw) >= 3 and "@" not in raw
    return len(raw) >= 2


def semantic_extract(doc, wanted, audit):
    """Find fields the fuzzy matchers missed by embedding the leading words of each CELL of each line and
    comparing them with all known label phrases. Only `wanted` keys can be added (the expected fields of the
    candidate type that are still missing), the best-scoring key over ALL keys must be a wanted one with a clear
    margin, and the value must pass a type guard."""
    from .embed_classifier import get_embedder
    emb = get_embedder()
    if emb is None or not wanted:
        return []
    taken = [f.bbox for f in doc.fields]
    cands = []    # (label_words, value_words, label_text)
    for ln in _lines(doc):
        cells = split_cells(ln)
        for ci, cell in enumerate(cells):
            for n in range(1, min(4, len(cell)) + 1):
                rest = cell[n:]
                vw = rest if rest else (cells[ci + 1] if ci + 1 < len(cells) else [])
                if vw and vw[0].text == ":":
                    vw = vw[1:]
                if not vw or (rest and rest[0].text.endswith(":")):
                    continue
                txt = " ".join(w.text for w in cell[:n]).strip(" :")
                if not re.search(r"[A-Za-z]{3}", txt) or _bad_label(txt) or _overlaps(_union(vw), taken):
                    continue
                cands.append((cell[:n], vw, txt))
    if not cands:
        return []
    keys, P = _label_protos(emb)
    V = emb.embed([c[2] for c in cands])
    S = V @ P.T
    best = {}      # key -> (sim, label_words, value_words)
    for ci, (lw, vw, txt) in enumerate(cands):
        per_key = {}
        for kj, k in enumerate(keys):
            per_key[k] = max(per_key.get(k, -1), float(S[ci, kj]))
        order = sorted(per_key.items(), key=lambda kv: -kv[1])
        (k1, s1), (k2, s2) = order[0], order[1]
        if k1 in wanted and s1 >= SEM_LABEL_MIN and s1 - s2 >= SEM_LABEL_MARGIN:
            if k1 not in best or s1 > best[k1][0]:
                best[k1] = (s1, lw, vw)
    added = []
    for k, (sim, lw, vw) in best.items():
        raw = " ".join(w.text for w in vw)
        bb = _union(vw)
        fin = finalize_value(k, raw)
        if fin is None or _overlaps(bb, [f.bbox for f in doc.fields]):
            continue
        raw, val = fin
        f = Field(key=k, label=" ".join(w.text for w in lw), raw=raw, value=val, bbox=bb,
                  conf=round(sum(w.conf for w in vw) / len(vw), 3), match="semantic", label_sim=round(sim, 3))
        doc.fields.append(f)
        added.append(f)
    if added:
        audit.log("semantic_fields", doc_id=doc.doc_id, added=[(f.key, f.label, f.label_sim) for f in added])
    return added


def lead_with_category(doc):
    """For an unsupported page that clearly resembles an unsupported class, say so first ('most similar to Certificate ...') instead of
    the field-count gate of the nearest SUPPORTED type, which reads as if the page were a bad form."""
    cat = doc.class_info.get("category")
    if doc.doc_type == "unknown" and cat and cat["key"].startswith("other:") and cat["confidence"] >= 0.4:
        why = doc.class_info.get("decision", "")
        lead = f"most similar to '{cat['label']}' (semantic confidence {cat['confidence']:.2f}), which is not a supported type"
        doc.class_info["decision"] = lead + (f" [{why}]" if why and not why.startswith("semantically closer") else "")


def classify(doc, audit):
    """Needs fields to be extracted first (structural evidence). Decision:
       1. pick a candidate type  (embeddings + keywords if a model is available, else keywords)
       2. optional semantic veto (page resembles a known unsupported class and keywords are weak)
       3. structural gate        (>= MIN_FIELDS of the candidate's expected fields actually extracted)"""
    from .embed_classifier import semantic_scores, SUPPORTED
    text = " ".join(w.text for w in doc.words)
    low = text.lower()
    kw = {t: round(len(keyword_hits(low, kws)) / len(kws), 2) for t, kws in DOC_TYPES.items()}
    from .id_patterns import STRONG_CUES
    for t, cues in STRONG_CUES.items():           # one cue alone is not enough (a Form 16 also says 'Income Tax Department')
        n_cues = len(keyword_hits(low, cues))
        has_id_pattern = bool({f.key for f in doc.fields} & {"aadhaar_no", "epic_no"}) or any(f.match == "pattern" and f.key == "id_number" for f in doc.fields)
        if n_cues >= 2 or (n_cues == 1 and has_id_pattern):
            kw[t] = max(kw[t], 0.8)
    found = {f.key for f in doc.fields}
    cov = {t: len(found & EXPECTED[t]) for t in EXPECTED}
    sem = semantic_scores(text)
    reason, cand, sem_ok = "", None, False
    if sem:
        sims = sem["sims"]
        hybrid = {t: sims[t] + KW_WEIGHT * kw[t] for t in SUPPORTED}
        guess = max(hybrid, key=hybrid.get)
        neg = max((t for t in sims if t.startswith("other:")), key=sims.get)
        if sims[neg] > sims[guess] + NEG_MARGIN and kw[guess] < 0.5 and cov.get(guess, 0) < VETO_MAX_FIELDS:
            from .embed_classifier import CATEGORY_NAMES
            reason = (f"semantically closer to '{CATEGORY_NAMES.get(neg, neg)}' ({sims[neg]:.2f}) than to "
                      f"'{CATEGORY_NAMES.get(guess, guess)}' ({sims[guess]:.2f}) and keyword evidence is weak")
        else:
            cand = guess
        sem_ok = sims[guess] >= sims[neg] + NEG_MARGIN   # page clearly resembles a supported type
    else:
        guess = max(kw, key=kw.get) if max(kw.values()) > 0 else "unknown"
        if guess != "unknown" and kw[guess] >= KW_CUTOFF:
            cand = guess
        else:
            reason = f"keyword score {kw.get(guess, 0)} < {KW_CUTOFF} (no embedding model active)"
    if cand and sem_ok and cov[cand] < len(EXPECTED[cand]):
        semantic_extract(doc, EXPECTED[cand] - found, audit)       # fill fields whose labels the dictionary lacks
        found = {f.key for f in doc.fields}
        cov = {t: len(found & EXPECTED[t]) for t in EXPECTED}
    strong = False
    if cand:
        if sem:
            strong = bool(sem_ok and kw[cand] >= STRONG_KW and cov[cand] >= 1
                          and max(SUPPORTED, key=sem["sims"].get) == cand)
        else:
            strong = kw[cand] >= STRONG_KW and cov[cand] >= 2
    if cand and cov[cand] < MIN_FIELDS and not strong:
        reason = f"only {cov[cand]} of {len(EXPECTED[cand])} expected '{cand}' fields found (need {MIN_FIELDS})"
        cand = None
    g = cand or guess
    ev = 0.5 * kw.get(g, 0) + 0.5 * (cov.get(g, 0) / len(EXPECTED[g]) if g in EXPECTED else 0)
    doc.best_guess = max(SUPPORTED, key=sem["sims"].get) if sem else guess     # display: embedding's own top pick
    doc.doc_type = cand or "unknown"
    if doc.doc_type in ("bank_statement", "salary_slip", "employment_letter"):      # 12-digit numbers there are account / employee numbers
        doc.fields = [f for f in doc.fields if f.key != "aadhaar_no"]
        doc.facts.pop("aadhaar_checksum_ok", None)
    doc.type_conf = round(ev, 2)
    from .embed_classifier import category_from_sims
    doc.class_info = {"keyword": kw, "expected_fields_found": cov, "semantic": sem, "category": (category_from_sims(sem["sims"]) if sem else None), "decision": reason or (f"accepted as {cand}" + ((f" (strong keyword + semantic evidence with {cov[cand]} field(s))" if sem else f" (strong keyword evidence with {cov[cand]} field(s), no embedding model)") if strong and cov[cand] < MIN_FIELDS else ""))}
    lead_with_category(doc)
    audit.log("classify", doc_id=doc.doc_id, doc_type=doc.doc_type, best_guess=guess, keyword=kw,
              expected_fields_found=cov, semantic=(sem or {}).get("sims"), decision=doc.class_info["decision"])


_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
DATE_PATTERNS = [
    r"\d{1,2} ?[/\-.] ?\d{1,2} ?[/\-.] ?\d{2,4}",           # '14- 03.1998': OCR spacing and mixed separators
    r"\d{4} ?[/\-.] ?\d{1,2} ?[/\-.] ?\d{1,2}",
    r"\d{1,2}[ \-/.]*" + _MON + r"[ \-/.,]*\d{2,4}",
]


def parse_date(s):
    t = s.strip()
    if re.search(_MON, t, re.I):                       # 14-MAR-1994, 14 March 1994, 14/Mar/94
        n = re.sub(r"[\s\-/.,]+", " ", t).strip().title().replace("Sept ", "Sep ")
        for fmt in ("%d %b %Y", "%d %B %Y", "%d %b %y", "%d %B %y"):
            try:
                return datetime.strptime(n, fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass
        return None
    t = re.sub(r"[/\-.]", "/", t.replace(" ", ""))        # one separator, so mixed '14-03.1998' parses
    for fmt in ("%d/%m/%Y", "%Y/%m/%d", "%d/%m/%y"):
        try:
            return datetime.strptime(t, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def parse_money(s):
    t = re.sub(r"(?i)rs\.?|inr|₹|,|\s", "", s)
    try:
        return float(t)
    except ValueError:
        pass
    if re.fullmatch(r"[0-9OlIL.]+", t) and any(c.isdigit() for c in t):      # OCR repair only when plain parsing failed
        try:
            return float(t.replace("O", "0").replace("l", "1").replace("I", "1").replace("L", "1"))
        except ValueError:
            pass
    return None


def _junk_tail(post):
    """What may follow a value: nothing, or a few short non-numeric OCR fragments separated by spaces ('Rs. 5,000 i i')."""
    if post == "":
        return True
    if not post[0].isspace():                          # '$577.3K', '14/03/1994x': the value runs into other text
        return False
    toks = post.split()
    def junk(t):                                        # short fragment; a digit is tolerated only inside symbol noise ('«7')
        return len(t) <= 3 and (not any(c.isdigit() for c in t) or any(not c.isalnum() for c in t))
    return len(toks) <= 3 and all(junk(t) for t in toks)


# ---- what kind of value is this? (used to report values that sit under the wrong label) ---------------------------
_PAN_LIKE = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
_EPIC_LIKE = re.compile(r"^[A-Z]{3}\d{7}$")
KIND_NAMES = {"date": "a date", "amount": "an amount", "id": "an ID / reference number", "text": "text"}
EXPECTED_KIND = {"dob": {"date"}, "issue_date": {"date"}, "id_number": {"id"}, "account_no": {"id", "amount"},
                 "name": {"text"}, "father_name": {"text"}, "employer": {"text"}, "address": {"text"}}
# keys whose repeated, different values on one page are an anomaly (single-valued fields)
DUP_STRICT = {"dob", "id_number", "aadhaar_no", "epic_no", "account_no", "ifsc"}
DUP_KEYS = DUP_STRICT | {"net_pay", "gross", "deductions", "declared_income", "name", "father_name", "employer"}


def value_kind(raw):
    """Coarse type of a raw value: date | amount | id | text | other."""
    t = (raw or "").strip()
    c = t.replace(" ", "")
    if not c:
        return "other"
    for pat in DATE_PATTERNS:
        m = re.match(pat, t, re.I)
        if m and parse_date(m.group(0)):
            return "date"
    if _PAN_LIKE.match(c.upper()) or _EPIC_LIKE.match(c.upper()):
        return "id"
    if re.fullmatch(r"(?i)(rs\.?|inr|₹)?[\d,]+(\.\d{1,2})?", c) and any(ch.isdigit() for ch in c):
        digits = re.sub(r"\D", "", c)
        return "id" if (len(digits) >= 10 and "," not in c and "." not in c) else "amount"
    if re.fullmatch(r"[A-Za-z0-9\-/]{6,}", c) and any(ch.isdigit() for ch in c) and any(ch.isalpha() for ch in c):
        return "id"
    letters = sum(ch.isalpha() for ch in t)
    return "text" if letters >= 4 and letters >= 0.6 * len(c) else "other"


def key_expected_kinds(key):
    if key in DATE_KEYS:
        return {"date"}
    if key in MONEY_KEYS:
        return {"amount"}
    return EXPECTED_KIND.get(key)


def is_misplaced(key, kind):
    """True when the value clearly belongs to a different kind of field than the label it was found under."""
    exp = key_expected_kinds(key)
    return bool(exp) and kind != "other" and kind not in exp


def _note_duplicate(doc, key, kept, raw, val, bbox, conf):
    """The same single-valued field was found a second time with a DIFFERENT value; the first one stays, the conflict is recorded."""
    norm = lambda v: re.sub(r"[^a-z0-9]", "", str(v).lower())
    if key in DUP_KEYS and kept is not None and val != kept.value and norm(val) != norm(kept.value):
        dups = doc.facts.setdefault("duplicates", [])
        if len(dups) < 12 and not any(d["key"] == key and d["other"]["raw"] == raw for d in dups):
            dups.append({"key": key, "kept": {"raw": kept.raw, "bbox": list(kept.bbox)},
                         "other": {"raw": raw[:40], "bbox": list(bbox), "conf": round(conf, 3)}})


LABEL_GUARD_KEYS = {"name", "father_name", "employer", "address", "branch", "account_type", "employment_type", "pay_month", "period",
                    "term", "currency"}


def _masked_number(raw):
    """'XXXX XXXX 4821' / '****4821' -> 'XXXXXXXX4821' (mask characters plus 3 to 8 visible digits); None when it is not a masked number."""
    compact = re.sub(r"[\s-]", "", (raw or "").strip())
    if re.fullmatch(r"(?i)[X*\u2022#]{2,}\d{3,8}", compact):
        return re.sub(r"[*\u2022#]", "X", compact.upper())
    return None


def looks_like_label(raw):
    """True when `raw` is itself a field label ('DATE OF BIRTH', 'MONTHLY GROSS SALARY'). In grid layouts the text next to or under a
    label is often another label; it must never be stored as the value."""
    t = (raw or "").strip(" :")
    if not re.search(r"[A-Za-z]{3}", t) or len(t.split()) > 6 or re.search(r"\d", t):      # labels carry no digits ('24 months' is a value)
        return False
    return _best_label(t)[0] is not None


def finalize_value(key, raw):
    """Single validation/normalisation step used by EVERY extraction pass. The value must START the string; only short
    trailing OCR junk is cut (never numbers taken from the middle of other text). Parses and applies the type guard.
    Returns (clean_raw, value) or None."""
    raw = raw.strip()
    if key in DATE_KEYS:
        for pat in DATE_PATTERNS:
            m = re.match(pat, raw, re.I)
            if m and _junk_tail(raw[m.end():]) and parse_date(m.group(0)):
                return m.group(0), parse_date(m.group(0))
        return None
    if key in MONEY_KEYS:
        m = re.match(r"(?i)(?:rs\.?|inr|₹)?\s*[0-9OlI][0-9OlI,]*(?:\.[0-9]{1,2})?", raw)
        if m and any(c.isdigit() for c in m.group(0)) and _junk_tail(raw[m.end():]) and parse_money(m.group(0)) is not None:
            return m.group(0).strip(), parse_money(m.group(0))
        return None
    if key == "ifsc":
        m = re.search(r"[A-Z]{4}0[A-Z0-9]{6}", raw.upper())
        return (m.group(0), m.group(0)) if m else None
    if key == "id_number":
        parts = raw.split(None, 1)
        first = parts[0].strip(",.;:") if parts else ""
        rest = raw[len(parts[0]):] if parts else ""
        if (len(re.sub(r"[^A-Za-z0-9]", "", first)) >= 6 and any(c.isdigit() for c in first)
                and (first.count("-") < 2 or re.search(r"[A-Za-z]-\d{6,}$", first)) and _junk_tail(rest)):
            return first, first.upper()
        joined = re.sub(r"\s", "", raw)                  # spaced digit groups (12-digit style IDs)
        return (raw, joined.upper()) if _value_ok(key, raw) and re.fullmatch(r"[0-9 ]{10,}", raw) else None
    if key == "account_no":
        masked = _masked_number(raw)
        if masked:                                       # statements and slips often print only the last digits ('XXXX XXXX 4821')
            return raw, masked
        return (raw, re.sub(r"\s", "", raw)) if _value_ok(key, raw) and re.fullmatch(r"[0-9 ]{6,}", raw) else None
    if key in ("employee_id", "case_ref"):                  # first token only; the rest must be short OCR junk
        parts = raw.split(None, 1)
        first = parts[0].strip(",.;:") if parts else ""
        if first and _value_ok(key, first) and _junk_tail(raw[len(parts[0]):]):
            return first, first.upper()
        return None
    if key in LABEL_GUARD_KEYS and looks_like_label(raw):  # a neighbouring label is never the value of this one
        return None
    return (raw, raw) if _value_ok(key, raw) else None


def _best_label(label_text):
    """Fuzzy label match. Short single-word labels must match almost exactly (>=90) because
    similar-looking words ('Education' vs 'deductions') otherwise create false fields."""
    lt = re.sub(r"[^a-z/ ]", "", label_text.lower()).strip()
    if not lt:
        return None, 0
    best, bs = None, 0
    for key, cands in LABELS.items():
        for c in cands:
            sc = fuzz.ratio(lt, c)
            need = 90 if (" " not in c or " " not in lt) else 84
            if sc >= need and sc > bs:
                best, bs = key, sc
    return (best, bs) if best else (None, 0)


def _lines(doc, min_conf=0.45):
    """Visual lines with leading OCR noise removed. Very-low-confidence words (e.g. Devanagari read by an English OCR)
    at the START of a line are dropped so that a real label such as 'Name:' is again the first word; words later in the
    line, which may be genuine but poorly read values, are always kept."""
    out = []
    for ln in group_lines(doc.words):
        i = 0
        while i < len(ln) and (ln[i].conf < min_conf or not (re.search(r"[A-Za-z0-9]", ln[i].text) or ln[i].conf >= 0.6)):
            i += 1
        if i < len(ln):
            out.append(ln[i:])
    return out


def split_cells(ln, factor=2.5):
    """Split a visual line into cells at large horizontal gaps (multi-column headers, label | value layouts)."""
    hs = sorted(w.bbox[3] - w.bbox[1] for w in ln)
    thr = max(factor * hs[len(hs) // 2], 45)
    cells = [[ln[0]]]
    for a, b in zip(ln, ln[1:]):
        if b.bbox[0] - a.bbox[2] > thr:
            cells.append([b])
        else:
            cells[-1].append(b)
    return cells


def _union(ws):
    return (min(w.bbox[0] for w in ws), min(w.bbox[1] for w in ws), max(w.bbox[2] for w in ws), max(w.bbox[3] for w in ws))


def _overlaps(bb, boxes):
    for o in boxes:
        if min(bb[2], o[2]) > max(bb[0], o[0]) and min(bb[3], o[3]) > max(bb[1], o[1]):
            return True
    return False


ANYWHERE_OK = {"ifsc"}      # distinctive single-word labels that may appear mid-cell


def extract_cells(doc, audit):
    """Second extraction pass for real-world layouts: labels in any cell of a line (e.g. 'ACCOUNT NO.' after an
    address column) or inside a cell after other text ('... BALANCE FORWARD'). The value is the rest of the cell,
    or the next cell. Only keys not found yet are added, values must pass the type guards, and single-word labels
    are accepted only at the start of a cell (multi-word labels may appear mid-cell)."""
    have = {f.key for f in doc.fields}
    taken = [f.bbox for f in doc.fields]
    added = []
    for ln in _lines(doc):
        cells = split_cells(ln)
        for ci, cell in enumerate(cells):
            if len(cell) >= 6 and cell[-1].text.endswith("."):          # a sentence ('The monthly gross salary amount excludes ...'), not a label | value cell
                continue
            for j in range(len(cell)):
                best = (None, 0, 0)
                for n in range(1, min(5, len(cell) - j) + 1):
                    txt = " ".join(w.text for w in cell[j:j + n]).strip(" :")
                    if not re.search(r"[A-Za-z]{3}", txt) or _bad_label(txt):
                        continue
                    key, sc = _best_label(txt)
                    if key and (sc > best[1] or (sc == best[1] and n > best[2])):
                        best = (key, sc, n)
                key, sc, n = best
                if not key or (key in have and key not in DUP_KEYS):
                    continue
                if j > 0 and n < 2 and key not in ANYWHERE_OK:
                    continue
                rest = cell[j + n:]
                vw = rest if rest else (cells[ci + 1] if ci + 1 < len(cells) else [])
                if vw and vw[0].text == ":":
                    vw = vw[1:]
                if not vw or (rest and rest[0].text.endswith(":")):
                    continue
                raw = " ".join(w.text for w in vw)
                bb = _union(vw)
                fin = finalize_value(key, raw)
                if key in have:                                           # a second occurrence of an already extracted field
                    if fin and not _overlaps(bb, taken):
                        _note_duplicate(doc, key, next((f for f in doc.fields if f.key == key), None), fin[0], fin[1], bb,
                                        sum(w.conf for w in vw) / len(vw))
                    continue
                if fin is None or _overlaps(bb, taken):
                    if fin is None:
                        kind = value_kind(raw)
                        if key in DATE_KEYS | MONEY_KEYS:
                            doc.facts.setdefault("unparsed", []).append({"key": key, "raw": raw[:40], "kind": kind, "bbox": list(bb)})
                        elif key in EXPECTED_KIND and is_misplaced(key, kind):
                            doc.facts.setdefault("rejected", []).append({"key": key, "raw": raw[:40], "kind": kind, "bbox": list(bb)})
                    continue
                raw, val = fin
                f = Field(key=key, label=" ".join(w.text for w in cell[j:j + n]).strip(" :,;"), raw=raw, value=val, bbox=bb,
                          conf=round(sum(w.conf for w in vw) / len(vw), 3), match="fuzzy-cell", label_sim=round(float(sc), 1))
                doc.fields.append(f)
                added.append(f)
                have.add(key)
                taken.append(bb)
    if added:
        audit.log("cell_fields", doc_id=doc.doc_id, added=[(f.key, f.label) for f in added])
    return added


def extract_stacked(doc, audit):
    """Label ABOVE value (key-value grids, form sheets): the label is a small cell and its value is the cell on the next line, aligned
    with it (left, right or centre). A money value may also sit to the right on the next line (a big 'NET PAY' figure beside its label).
    The label must be a whole cell, the value must pass the type guard and must not itself be a label. Only keys not found yet are
    added (a second, different value of an existing key is recorded as a duplicate). Failures are silent: table headers such as
    'DEDUCTIONS' have text below them that is not their value."""
    lines = _lines(doc)
    if len(lines) < 3:
        return []
    cells, by_line = [], {}
    for li, ln in enumerate(lines):
        for c in split_cells(ln):
            by_line.setdefault(li, []).append(len(cells))
            cells.append((li, c, _union(c), " ".join(w.text for w in c)))
    have = {f.key for f in doc.fields}
    taken = [f.bbox for f in doc.fields]
    added = []

    def aligned(cb, db, ch):
        tol = max(30.0, 1.5 * ch)
        return (abs(db[0] - cb[0]) <= tol or abs(db[2] - cb[2]) <= tol
                or abs((db[0] + db[2]) / 2 - (cb[0] + cb[2]) / 2) <= tol)

    for li, cw, cb, ctext in cells:
        txt = ctext.strip(" :")
        if len(cw) > 7 or not re.search(r"[A-Za-z]{3}", txt) or _bad_label(txt):
            continue
        key, sc = _best_label(txt)
        if not key:
            continue
        ch = cb[3] - cb[1]
        vw = None
        for lj in (li + 1, li + 2):
            cand = []
            for di in by_line.get(lj, []):
                _, dw, db, dtext = cells[di]
                dh = db[3] - db[1]
                gap = db[1] - cb[3]
                if gap < -0.5 * min(ch, dh) or gap > 2.6 * max(ch, dh) or not aligned(cb, db, ch):
                    continue
                if looks_like_label(dtext) or _overlaps(db, taken):
                    continue
                cand.append((abs(db[0] - cb[0]), di))
            if cand:
                vw = list(cells[min(cand)[1]][1])
                lastb = _union(vw)
                if key == "address":                                     # wrapped address: following lines left-aligned with the first
                    for lk in range(lj + 1, lj + 3):
                        nxt = [cells[di] for di in by_line.get(lk, [])
                               if abs(cells[di][2][0] - lastb[0]) <= max(30.0, 1.5 * ch) and 0 <= cells[di][2][1] - lastb[3] <= 0.9 * (lastb[3] - lastb[1])
                               and not looks_like_label(cells[di][3])]
                        if not nxt:
                            break
                        vw += list(nxt[0][1])
                        lastb = _union(nxt[0][1])
                break
        if vw is None and key in MONEY_KEYS:                             # figure to the right of its label, on the next line
            cand = []
            for di in by_line.get(li + 1, []):
                _, dw, db, dtext = cells[di]
                dh = db[3] - db[1]
                if db[0] >= cb[2] - 30 and -0.6 * ch <= db[1] - cb[3] <= 1.3 * dh and not looks_like_label(dtext):
                    cand.append((db[0] - cb[2], di))
            if cand:
                vw = list(cells[min(cand)[1]][1])
        if not vw:
            continue
        raw = " ".join(w.text for w in vw)
        bb = _union(vw)
        fin = finalize_value(key, raw)
        if fin is None:
            continue
        if key in have:
            if not _overlaps(bb, taken):
                _note_duplicate(doc, key, next((f for f in doc.fields if f.key == key), None), fin[0], fin[1], bb,
                                sum(w.conf for w in vw) / len(vw))
            continue
        raw, val = fin
        f = Field(key=key, label=txt, raw=raw, value=val, bbox=bb, conf=round(sum(w.conf for w in vw) / len(vw), 3),
                  match="stacked", label_sim=round(float(sc), 1))
        doc.fields.append(f)
        added.append(f)
        have.add(key)
        taken.append(bb)
    if added:
        audit.log("stacked_fields", doc_id=doc.doc_id, added=[(f.key, f.label) for f in added])
    return added


def mask_aadhaar(digits):
    d = re.sub(r"\D", "", digits)
    return "XXXX XXXX " + d[-4:] if len(d) == 12 else "XXXX"


_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DT = r"(?:\d{1,2}[-/. ](?:" + _MON + r"|\d{1,2})[-/. ,]*\d{2,4})"
TITLE_MONTH = re.compile(r"(?i)\b(?:salary\s+slip|pay\s*slip|salary\s+statement|pay\s+statement|payroll)\b[\s:,\-]*(?:for|of|month of)?[\s:,\-]*"
                         r"(?P<v>" + _MON + r"\s*,?\s*(?:19|20)\d{2})")
TITLE_RANGE = re.compile(r"(?i)\bstatement\b[\s:,\-]*(?:period|date)?[\s:,\-]*(?P<v>" + _DT + r"\s*(?:to|-|\u2013|\u2014)\s*" + _DT + r")")


def extract_context_fields(doc, audit):
    """Fields stated inside a title line rather than next to a label: 'Salary Slip for October 2026' (pay month) and
    'Account Statement: 01-Oct-2026 to 31-Oct-2026' (statement period). Only added when no labelled field of that key exists."""
    have = {f.key for f in doc.fields}
    added = []
    for ln in _lines(doc):
        text = " ".join(w.text for w in ln)
        for key, rx in (("pay_month", TITLE_MONTH), ("period", TITLE_RANGE)):
            if key in have:
                continue
            m = rx.search(text)
            if not m:
                continue
            raw = m.group("v").strip()
            if not _value_ok(key, raw):
                continue
            pos, ws = 0, []
            for w in ln:                                          # words overlapping the matched span
                lo, hi = pos, pos + len(w.text)
                if hi > m.start("v") and lo < m.end("v"):
                    ws.append(w)
                pos = hi + 1
            if not ws:
                continue
            bbox = (min(w.bbox[0] for w in ws), min(w.bbox[1] for w in ws), max(w.bbox[2] for w in ws), max(w.bbox[3] for w in ws))
            doc.fields.append(Field(key=key, label="title line", raw=raw, value=raw, bbox=bbox,
                                    conf=round(sum(w.conf for w in ws) / len(ws), 3), match="title", label_sim=100.0))
            have.add(key)
            added.append(key)
    if added:
        audit.log("context_fields", doc_id=doc.doc_id, added=added)


def extract_id_patterns(doc, audit):
    """Label-free extraction for Indian ID documents (see id_patterns.py). Adds, only when missing:
    aadhaar_no (kept separate from the PAN-style id_number so different ID kinds are never compared),
    id_number (PAN pattern), name (line above DOB), address (up to the PIN code). The Aadhaar number is stored
    masked in `raw`; the digits stay in `value` for matching, and the Verhoeff checksum result in doc.facts."""
    from .id_patterns import find_aadhaar, find_pan, find_epic, pan_card_fields, name_above_dob, address_to_pin
    lines = group_lines(doc.words)
    have = {f.key for f in doc.fields}
    added = []

    def add(key, label, raw, value, hit, sim=1.0):
        doc.fields.append(Field(key=key, label=label, raw=raw, value=value, bbox=tuple(hit["bbox"]), conf=hit["conf"],
                                match="pattern", label_sim=sim))
        have.add(key)
        added.append(key)
    if "aadhaar_no" not in have:
        owned = [f.bbox for f in doc.fields if f.key in ("account_no", "id_number")]      # a 12-digit account number is not an Aadhaar
        for a in find_aadhaar(lines):
            if _overlaps(tuple(a["bbox"]), owned):
                continue
            add("aadhaar_no", "Aadhaar number", mask_aadhaar(a["digits"]), a["digits"], a)
            doc.facts["aadhaar_checksum_ok"] = bool(a["checksum_ok"])
            break
    if "epic_no" not in have:                    # Voter ID number: kept separate from PAN/Aadhaar so ID kinds are never compared
        for e in find_epic(lines):
            add("epic_no", "EPIC pattern", e["raw"], e["raw"], e)
            break
    if "id_number" not in have:
        for pan in find_pan(lines):
            add("id_number", "PAN pattern", pan["raw"], pan["raw"], pan)
            break
    if any(f.key == "id_number" and f.match == "pattern" for f in doc.fields):      # PAN card without labels
        lay = pan_card_fields(_lines(doc))
        for key in ("name", "father_name", "dob"):
            hit = lay.get(key)
            if not hit or key in have:
                continue
            if key == "dob":
                fin = finalize_value("dob", hit["raw"])
                if fin:
                    add("dob", "date on its own line", fin[0], fin[1], hit, 0.8)
            elif _value_ok("name", hit["raw"]):
                add(key, "stacked layout", hit["raw"], hit["raw"], hit, 0.8)
    if "name" not in have:
        n = name_above_dob(lines)
        if n and _value_ok("name", n["raw"]):
            add("name", "line above DOB", n["raw"], n["raw"], n, 0.8)
    if "address" not in have:
        a = address_to_pin(lines)
        if a and _value_ok("address", a["raw"]):
            add("address", "Address (to PIN code)", a["raw"], a["raw"], a)
    if added:
        audit.log("pattern_fields", doc_id=doc.doc_id, added=added)
    return added


def _split_qualifier(words):
    """(qualifier_words, rest): a bracketed note right after a label ('(PF+Tax)', '(last 4)', '(INR)') is part of the label, not of the value.
    Brackets without letters ('(10,000)' = negative amount) stay in the value."""
    if not words or not words[0].text.startswith("("):
        return [], words
    for i in range(min(len(words), 5)):
        if ")" in words[i].text:
            inner = " ".join(w.text for w in words[:i + 1])
            if re.search(r"[A-Za-z]{2}", inner) and words[i].text.rstrip(".,:;").endswith(")"):
                return list(words[:i + 1]), list(words[i + 1:])
            break
    return [], words


def extract_fields(doc, audit):
    lines = _lines(doc)
    fields = []
    for ln in lines:
        # label = best-matching leading words of the line (colon optional: OCR often drops it)
        best = (None, 0, 0)  # key, score, n_words
        for n in range(1, min(5, len(ln) - 1) + 1):
            cand = " ".join(w.text for w in ln[:n])
            key, sc = _best_label(cand)
            if key and (sc > best[1] or (sc == best[1] and n > best[2])):
                best = (key, sc, n)
        key, sc, n = best
        if not key:
            continue
        label_words, value_words = ln[:n], ln[n:]
        if _bad_label(" ".join(w.text for w in label_words)):
            continue
        qual, rest = _split_qualifier(value_words)                 # 'Deductions (PF+Tax) 10,000': the bracket belongs to the label
        if qual and rest:
            label_words, value_words = label_words + qual, rest
        # a leading ':' glued to the first value token
        if value_words and value_words[0].text == ":":
            value_words = value_words[1:]
        if not value_words:
            continue
        if value_words[0].text.endswith(":"):      # label continues ('Company Name:'): not a clean label/value split
            continue
        raw0 = " ".join(w.text for w in value_words)
        fin = finalize_value(key, raw0)
        if fin is None:
            kind = value_kind(raw0)
            vb = list(_union(value_words))
            if key in DATE_KEYS | MONEY_KEYS | {"id_number", "account_no"}:     # label found but value unreadable: report, do not store
                doc.facts.setdefault("unparsed", []).append({"key": key, "raw": raw0[:40], "kind": kind, "bbox": vb})
            elif key in EXPECTED_KIND and is_misplaced(key, kind):               # e.g. a date under 'Name'
                doc.facts.setdefault("rejected", []).append({"key": key, "raw": raw0[:40], "kind": kind, "bbox": vb})
            continue
        raw, val = fin
        if raw != raw0:
            doc.facts.setdefault("ocr_cleaned", []).append({"key": key, "original": raw0[:40], "cleaned": raw})
        bbox = (min(w.bbox[0] for w in value_words), min(w.bbox[1] for w in value_words),
                max(w.bbox[2] for w in value_words), max(w.bbox[3] for w in value_words))
        conf = sum(w.conf for w in value_words) / len(value_words)
        fields.append(Field(key=key, label=" ".join(w.text for w in label_words),
                            raw=raw, value=val, bbox=bbox, conf=round(conf, 3),
                            match="fuzzy", label_sim=round(float(sc), 1)))
    # keep first occurrence of each key
    seen, uniq, first = set(), [], {}
    for f in fields:
        if f.key not in seen:
            seen.add(f.key)
            uniq.append(f)
            first[f.key] = f
        else:                                              # same key again: keep the first, but report a conflicting second value
            _note_duplicate(doc, f.key, first[f.key], f.raw, f.value, f.bbox, f.conf)
    doc.fields = uniq
    extract_stacked(doc, audit)
    extract_cells(doc, audit)
    extract_id_patterns(doc, audit)
    extract_context_fields(doc, audit)
    doc.status = "extracted"
    audit.log("extract", doc_id=doc.doc_id, fields=[f.key for f in doc.fields])
    return doc.fields


ISSUER_TYPES = {"salary_slip", "employment_letter"}
TITLE_WORDS = {"salary", "slip", "payslip", "pay", "statement", "bank", "account", "employment", "income", "confirmation", "certificate", "letter",
               "form", "application", "credit", "personal", "identity", "record", "declaration", "self", "declared", "of", "for", "the", "and",
               "transaction", "monthly", "payroll", "earnings", "verification", "internal", "desk", "ledger", "report", "invoice", "receipt"}
CORP_CUE = re.compile(r"(?i)\b(pvt|ltd|limited|llp|inc|corp|corporation|company|technologies|solutions|services|analytics|industries|systems|enterprises|bank|finance|labs)\b")


def derive_issuer(doc):
    """The organisation named in the page header: the tallest text cell in the top fifth of the page. Returns (text, bbox, conf) or None."""
    best = None
    for ln in group_lines(doc.words):
        if (ln[0].bbox[1] + ln[0].bbox[3]) / 2 > 0.22 * doc.height:
            break
        for cell in split_cells(ln):
            txt = " ".join(w.text for w in cell).strip()
            conf = sum(w.conf for w in cell) / len(cell)
            if sum(c.isalpha() for c in txt) < 6 or len(cell) > 8 or conf < 0.7 or looks_like_label(txt) or sum(c.isdigit() for c in txt) > 3:
                continue
            if all(re.sub(r"[^a-z]", "", w.text.lower()) in TITLE_WORDS for w in cell):      # 'SALARY SLIP', 'Bank Statement': a document title, not an organisation
                continue
            h = sorted(w.bbox[3] - w.bbox[1] for w in cell)[len(cell) // 2]
            if best is None or h > best[0] * 1.05:
                best = (h, txt, _union(cell), conf)
    return None if best is None else (best[1], best[2], best[3])


def post_classify(doc, audit):
    """Steps that need the document type. (1) An application states its income as 'monthly gross salary': it is the declared income.
    (2) A payslip or employment letter that does not print its employer as a field is issued BY the employer: the organisation named
    in the page header is used (marked match='header', so it can be told from a labelled field). The issuer is also kept in doc.facts."""
    if doc.doc_type == "application_form" and doc.get("monthly_gross") and not doc.get("declared_income"):
        doc.get("monthly_gross").key = "declared_income"
    iss = derive_issuer(doc) if doc.doc_type != "unknown" else None
    if iss:
        doc.facts["issuer"] = {"text": iss[0], "bbox": list(iss[1]), "conf": round(iss[2], 3)}


def issuer_as_employer(docs, audit):
    """Late step (after the visual checks, so no pixel check ever treats it as a printed field): a payslip or employment letter that does not
    print its employer as a field is issued BY the employer, so the organisation named in its page header becomes the `employer` field,
    marked match='header'. Used for generated PDFs (exact text) or, on scans, only when the header reads like a company name."""
    for d in docs:
        iss = d.facts.get("issuer")
        if not iss or d.doc_type not in ISSUER_TYPES or d.get("employer") or not _value_ok("employer", iss["text"]):
            continue
        if not d.born_digital and not (CORP_CUE.search(iss["text"]) and iss.get("conf", 0) >= 0.8):
            continue
        d.fields.append(Field(key="employer", label="issuer (page header)", raw=iss["text"], value=iss["text"], bbox=tuple(iss["bbox"]),
                              conf=iss.get("conf", 0.8), match="header", label_sim=0.6))
        audit.log("issuer_as_employer", doc_id=d.doc_id, employer=iss["text"])


def assign_file_categories(docs, audit):
    """File-level category: pages of the same source file vote (weighted by text amount and how decisive each page is).
    Gives every page `class_info['file_category']`, so an unsupported file is described ('Corporate financial report, 0.73')
    instead of only being called 'unknown'. Needs the semantic model; without it nothing is added."""
    from .embed_classifier import category_from_pages
    groups = {}
    for d in docs:
        groups.setdefault(d.path, []).append(d)
    for path, ds in groups.items():
        rows = [((d.class_info.get("semantic") or {}).get("sims"), min(len(d.words), 400)) for d in ds if len(d.words) >= 10]
        rows = [r for r in rows if r[0]]
        if not rows:
            continue
        cat = category_from_pages([r[0] for r in rows], [r[1] for r in rows])
        for d in ds:
            d.class_info["file_category"] = cat
        audit.log("categorise", file=os.path.basename(path), label=cat["label"], confidence=cat["confidence"],
                  pages_agree=cat["pages_agree"], top=[(t["name"], t["confidence"]) for t in cat["top"]])


def _repeats_table_header(d, T):
    """True when the page carries the table header typical of type T (transaction table for a bank statement, earnings/deductions for a payslip)."""
    want = {"bank_statement": "transactions", "salary_slip": "components"}.get(T)
    if not want or not d.words:
        return False
    try:
        from .structure import _find_headers
        headers, _ = _find_headers(group_lines(d.words))
    except Exception:
        return False
    return any(h["kind"] == want for h in headers)


def inherit_continuations(docs, audit):
    """Multi-page files: a page with no recognisable structure (e.g. transaction rows) inherits the type of the
    preceding recognised page of the SAME file when nothing contradicts it: no other supported type has real keyword
    support, the embedding (if active) still prefers that type and does not resemble an unsupported class."""
    from .embed_classifier import SUPPORTED
    by_src = {}
    for d in docs:
        by_src.setdefault(d.path, []).append(d)
    for ds in by_src.values():
        anchor = None
        for d in ds:
            if d.doc_type != "unknown":
                if not d.class_info.get("continuation_of"):
                    anchor = d
                continue
            if anchor is None:
                continue
            same_id = any(f.key in ("epic_no", "aadhaar_no", "id_number") and anchor.get(f.key) is not None
                          and str(anchor.get(f.key).value) == str(f.value) for f in d.fields)      # the back of a card repeats the number
            if len(d.fields) > 2 and not same_id:
                continue
            T, ci = anchor.doc_type, d.class_info
            kw = ci.get("keyword", {})
            others = [v for t, v in kw.items() if t != T]
            ok = kw.get(T, 0) >= 0.1 and (max(others) if others else 0) < 0.5
            sims = (ci.get("semantic") or {}).get("sims")
            if ok and sims and not _repeats_table_header(d, T):        # a repeated table header is stronger evidence than the semantic margin
                neg = max(v for k, v in sims.items() if k.startswith("other:"))
                ok = max(SUPPORTED, key=sims.get) == T and sims[T] >= neg + 0.05
            if same_id:
                ok = True
            if ok:
                d.doc_type = T
                for f in d.fields:                       # a 'balance forward' row on a later page is a carry-forward balance
                    if f.key == "opening_balance":
                        f.key = "balance_forward"
                ci["continuation_of"] = anchor.doc_id
                ci["decision"] = (f"continuation of {anchor.doc_id} ({T}): same ID number on both pages" if same_id else
                                  f"continuation of {anchor.doc_id} ({T}): no competing type and no unsupported-class match")
                audit.log("continuation", doc_id=d.doc_id, anchor=anchor.doc_id, doc_type=T)
