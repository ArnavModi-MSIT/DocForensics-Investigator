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
}

# ---- label dictionary per field key (fuzzy matched) -------------------------
LABELS = {
    "name": ["applicant name", "name", "account holder", "employee name", "holder name", "electors name", "elector name"],
    "dob": ["date of birth", "date of birth / age", "date of birth age", "date of birth/age", "dob", "birth date"],
    "father_name": ["fathers name", "father name", "father's name", "husbands name", "husband name"],
    "id_number": ["id number", "pan", "id no", "pan number"],
    "address": ["address", "residential address"],
    "employer": ["employer", "company", "organisation"],
    "declared_income": ["declared monthly income", "monthly income", "declared income"],
    "issue_date": ["issue date", "date of issue"],
    "gross": ["gross earnings", "gross salary", "gross pay"],
    "deductions": ["total deductions", "deductions"],
    "net_pay": ["net pay", "net salary", "take home"],
    "pay_month": ["pay month", "month", "salary month"],
    "account_no": ["account number", "account no", "a/c no"],
    "period": ["statement period", "statement date", "period"],
    "opening_balance": ["opening balance", "balance forward", "balance brought forward", "balance b/f"],
    "salary_credit": ["salary credit", "salary credited"],
    "closing_balance": ["closing balance"],
    "currency": ["currency"],
    "branch": ["branch"],
    "ifsc": ["ifsc", "ifsc code"],
    "account_type": ["account type"],
}
DATE_KEYS = {"dob", "issue_date"}
MONEY_KEYS = {"declared_income", "gross", "deductions", "net_pay",
              "opening_balance", "salary_credit", "closing_balance"}


EXPECTED = {   # fields that genuinely belong to each type (used as structural evidence)
    "application_form": {"name", "dob", "id_number", "address", "employer", "declared_income"},
    "id_card": {"name", "father_name", "dob", "id_number", "aadhaar_no", "epic_no", "address", "issue_date"},
    "salary_slip": {"name", "employer", "pay_month", "gross", "deductions", "net_pay"},
    "bank_statement": {"name", "account_no", "period", "opening_balance", "salary_credit", "closing_balance",
                       "currency", "branch", "ifsc", "account_type"},
}
MIN_FIELDS = int(os.environ.get("DOCFOR_MIN_FIELDS", 3))          # a page is a supported document only if >= this many of its expected fields were found
STRONG_KW = float(os.environ.get("DOCFOR_STRONG_KW", 0.75))   # keyword score that counts as strong evidence
KW_CUTOFF = 0.5         # keyword evidence needed when no embedding model is available
KW_WEIGHT = float(os.environ.get("DOCFOR_KW_WEIGHT", 0.2))         # weight of keywords when mixing with embedding similarity (tunable)
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
        return len(re.sub(r"\s", "", raw)) >= 6 and any(c.isdigit() for c in raw) and raw.count("-") < 2
    if key == "account_no":
        return sum(c.isdigit() for c in raw) >= 6
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
        if sims[neg] > sims[guess] + NEG_MARGIN and kw[guess] < 0.5:
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
    doc.type_conf = round(ev, 2)
    from .embed_classifier import category_from_sims
    doc.class_info = {"keyword": kw, "expected_fields_found": cov, "semantic": sem, "category": (category_from_sims(sem["sims"]) if sem else None), "decision": reason or (f"accepted as {cand}" + ((f" (strong keyword + semantic evidence with {cov[cand]} field(s))" if sem else f" (strong keyword evidence with {cov[cand]} field(s), no embedding model)") if strong and cov[cand] < MIN_FIELDS else ""))}
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
        if (len(re.sub(r"[^A-Za-z0-9]", "", first)) >= 6 and any(c.isdigit() for c in first) and first.count("-") < 2
                and _junk_tail(rest)):
            return first, first.upper()
        joined = re.sub(r"\s", "", raw)                  # spaced digit groups (12-digit style IDs)
        return (raw, joined.upper()) if _value_ok(key, raw) and re.fullmatch(r"[0-9 ]{10,}", raw) else None
    if key == "account_no":
        return (raw, re.sub(r"\s", "", raw)) if _value_ok(key, raw) and re.fullmatch(r"[0-9 ]{6,}", raw) else None
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
                if not key or key in have:
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
                if fin is None or _overlaps(bb, taken):
                    if fin is None and key in DATE_KEYS | MONEY_KEYS:
                        doc.facts.setdefault("unparsed", []).append({"key": key, "raw": raw[:40]})
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


def mask_aadhaar(digits):
    d = re.sub(r"\D", "", digits)
    return "XXXX XXXX " + d[-4:] if len(d) == 12 else "XXXX"


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
        for a in find_aadhaar(lines):
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
            if key in DATE_KEYS | MONEY_KEYS | {"id_number", "account_no"}:     # label found but value unreadable: report, do not store
                doc.facts.setdefault("unparsed", []).append({"key": key, "raw": raw0[:40]})
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
    seen, uniq = set(), []
    for f in fields:
        if f.key not in seen:
            seen.add(f.key)
            uniq.append(f)
    doc.fields = uniq
    extract_cells(doc, audit)
    extract_id_patterns(doc, audit)
    doc.status = "extracted"
    audit.log("extract", doc_id=doc.doc_id, fields=[f.key for f in doc.fields])
    return doc.fields


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
            if ok and sims:
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
