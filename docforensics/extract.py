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
    "name": ["applicant name", "name", "account holder", "employee name", "holder name"],
    "dob": ["date of birth", "dob", "birth date"],
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
    "period": ["statement period", "period"],
    "opening_balance": ["opening balance"],
    "salary_credit": ["salary credit", "salary credited"],
    "closing_balance": ["closing balance"],
}
DATE_KEYS = {"dob", "issue_date"}
MONEY_KEYS = {"declared_income", "gross", "deductions", "net_pay",
              "opening_balance", "salary_credit", "closing_balance"}


EXPECTED = {   # fields that genuinely belong to each type (used as structural evidence)
    "application_form": {"name", "dob", "id_number", "address", "employer", "declared_income"},
    "id_card": {"name", "dob", "id_number", "address", "issue_date"},
    "salary_slip": {"name", "employer", "pay_month", "gross", "deductions", "net_pay"},
    "bank_statement": {"name", "account_no", "period", "opening_balance", "salary_credit", "closing_balance"},
}
MIN_FIELDS = int(os.environ.get("DOCFOR_MIN_FIELDS", 3))          # a page is a supported document only if >= this many of its expected fields were found
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


def _value_ok(key, raw):
    """Type guard: the value must look like the field it is assigned to."""
    raw = raw.strip()
    if key in DATE_KEYS:
        return parse_date(raw) is not None
    if key in MONEY_KEYS:
        return parse_money(raw) is not None
    if key == "id_number":
        return len(re.sub(r"\s", "", raw)) >= 6 and any(c.isdigit() for c in raw)
    if key == "account_no":
        return sum(c.isdigit() for c in raw) >= 6
    if key == "period":                              # a date, a date range, or month + year
        return bool(DATE_LIKE.search(raw) or (MONTH_RE.search(raw) and YEAR_RE.search(raw)))
    if key == "pay_month":                           # month name, month/year or a year
        return bool(MONTH_RE.search(raw) or MONTH_NUM.search(raw) or YEAR_RE.search(raw))
    if key == "name":                                # person/company-like: few words, some capitalised, mostly letters
        toks = raw.split()
        t = re.sub(r"\s", "", raw)
        return (1 <= len(toks) <= 6 and len(t) >= 2 and "@" not in raw
                and sum(c.isdigit() for c in t) / len(t) < 0.2 and any(w[0].isupper() for w in toks))
    if key == "address":
        return len(raw) >= 8 and (any(c.isdigit() for c in raw) or "," in raw)
    if key == "employer":
        return sum(c.isalpha() for c in raw) >= 3 and "@" not in raw
    return len(raw) >= 2


def semantic_extract(doc, wanted, audit):
    """Find fields the fuzzy matcher missed by embedding the leading words of each line and
    comparing them with all known label phrases. Only `wanted` keys can be added (the expected
    fields of the candidate type that are still missing), the best-scoring key over ALL keys must
    be a wanted one with a clear margin, and the value must pass a type guard."""
    from .embed_classifier import get_embedder
    emb = get_embedder()
    if emb is None or not wanted:
        return []
    have_y = [((f.bbox[1] + f.bbox[3]) / 2) for f in doc.fields]
    cands = []   # (line_idx, n, label_text)
    lines = group_lines(doc.words)
    for li, ln in enumerate(lines):
        cy = sum((w.bbox[1] + w.bbox[3]) / 2 for w in ln) / len(ln)
        if any(abs(cy - y) < (ln[0].bbox[3] - ln[0].bbox[1]) * 0.6 for y in have_y):
            continue                                   # line already produced a field
        for n in range(1, min(4, len(ln) - 1) + 1):
            if ln[n].text.endswith(":"):               # label continues into the next token
                continue
            cands.append((li, n, " ".join(w.text for w in ln[:n]).strip(" :")))
    cands = [c for c in cands if re.search(r"[A-Za-z]{3}", c[2]) and not _bad_label(c[2])]
    if not cands:
        return []
    keys, P = _label_protos(emb)
    V = emb.embed([c[2] for c in cands])
    S = V @ P.T
    best = {}      # key -> (sim, line_idx, n)
    for ci, (li, n, txt) in enumerate(cands):
        per_key = {}
        for kj, k in enumerate(keys):
            per_key[k] = max(per_key.get(k, -1), float(S[ci, kj]))
        order = sorted(per_key.items(), key=lambda kv: -kv[1])
        (k1, s1), (k2, s2) = order[0], order[1]
        if k1 in wanted and s1 >= SEM_LABEL_MIN and s1 - s2 >= SEM_LABEL_MARGIN:
            if k1 not in best or s1 > best[k1][0]:
                best[k1] = (s1, li, n)
    added = []
    for k, (sim, li, n) in best.items():
        ln = lines[li]
        vw = ln[n:]
        if vw and vw[0].text == ":":
            vw = vw[1:]
        if not vw:
            continue
        raw = " ".join(w.text for w in vw)
        if not _value_ok(k, raw):
            continue
        bbox = (min(w.bbox[0] for w in vw), min(w.bbox[1] for w in vw), max(w.bbox[2] for w in vw), max(w.bbox[3] for w in vw))
        val = raw
        if k in DATE_KEYS:
            val = parse_date(raw) or raw
        elif k in MONEY_KEYS:
            val = parse_money(raw)
        elif k == "id_number":
            val = re.sub(r"\s", "", raw).upper()
        f = Field(key=k, label=" ".join(w.text for w in ln[:n]), raw=raw, value=val, bbox=bbox,
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
            reason = f"semantically closer to '{neg}' ({sims[neg]:.2f}) than to '{guess}' ({sims[guess]:.2f}) and keyword evidence is weak"
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
    if cand and cov[cand] < MIN_FIELDS:
        reason = f"only {cov[cand]} of {len(EXPECTED[cand])} expected '{cand}' fields found (need {MIN_FIELDS})"
        cand = None
    g = cand or guess
    ev = 0.5 * kw.get(g, 0) + 0.5 * (cov.get(g, 0) / len(EXPECTED[g]) if g in EXPECTED else 0)
    doc.best_guess = max(SUPPORTED, key=sem["sims"].get) if sem else guess     # display: embedding's own top pick
    doc.doc_type = cand or "unknown"
    doc.type_conf = round(ev, 2)
    doc.class_info = {"keyword": kw, "expected_fields_found": cov, "semantic": sem, "decision": reason or f"accepted as {cand}"}
    audit.log("classify", doc_id=doc.doc_id, doc_type=doc.doc_type, best_guess=guess, keyword=kw,
              expected_fields_found=cov, semantic=(sem or {}).get("sims"), decision=doc.class_info["decision"])


def parse_date(s):
    s = s.strip().replace(" ", "")
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def parse_money(s):
    t = re.sub(r"(?i)rs\.?|inr|₹|,|\s", "", s)
    t = t.replace("O", "0").replace("l", "1") if re.fullmatch(r"[0-9OlL.]+", t) else t
    try:
        return float(t)
    except ValueError:
        return None


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


def extract_fields(doc, audit):
    lines = group_lines(doc.words)
    fields = []
    for ln in lines:
        # label = best-matching leading words of the line (colon optional: OCR often drops it)
        best = (None, 0, 0)  # key, score, n_words
        for n in range(1, min(4, len(ln) - 1) + 1):
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
        raw = " ".join(w.text for w in value_words)
        if key in ("period", "pay_month") and not _value_ok(key, raw):
            continue                                   # loosely typed keys: require a plausible value
        bbox = (min(w.bbox[0] for w in value_words), min(w.bbox[1] for w in value_words),
                max(w.bbox[2] for w in value_words), max(w.bbox[3] for w in value_words))
        conf = sum(w.conf for w in value_words) / len(value_words)
        val = raw
        if key in DATE_KEYS:
            val = parse_date(raw) or raw
        elif key in MONEY_KEYS:
            m = parse_money(raw)
            val = m if m is not None else raw
        elif key == "id_number":
            val = re.sub(r"\s", "", raw).upper()
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
    doc.status = "extracted"
    audit.log("extract", doc_id=doc.doc_id, fields=[f.key for f in uniq])
    return uniq
