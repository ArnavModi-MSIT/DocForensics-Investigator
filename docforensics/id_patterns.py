"""Label-free extractors for Indian ID documents (Aadhaar, PAN) + Verhoeff checksum.

Pure functions over OCR words/lines (objects with .text, .bbox=(x0,y0,x1,y1), .conf), so they can be
used from docforensics/extract.py without touching the rest of the pipeline.

Privacy note: nothing here stores or logs numbers; callers decide what to keep.
"""
import re

# ---------------------------------------------------------------- Verhoeff checksum (used by Aadhaar)
_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
      [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
      [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
      [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
      [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(number: str) -> bool:
    """True if the digit string (check digit included) passes the Verhoeff check."""
    if not number.isdigit():
        return False
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(number: str) -> int:
    """Check digit to append to `number` (used only for building test fixtures)."""
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return _INV[c]


# ---------------------------------------------------------------- helpers
def _union(ws):
    return (min(w.bbox[0] for w in ws), min(w.bbox[1] for w in ws), max(w.bbox[2] for w in ws), max(w.bbox[3] for w in ws))


def _conf(ws):
    return round(sum(w.conf for w in ws) / len(ws), 3)


# ---------------------------------------------------------------- Aadhaar number (4-4-4 digits, no label needed)
def find_aadhaar(lines):
    """Find 12-digit Aadhaar-style numbers: three 4-digit tokens in a row on one line (or one 12-digit token).
    Excludes longer digit runs (e.g. the 16-digit Virtual ID). Returns list of dicts:
    {raw, digits, bbox, conf, checksum_ok}. `checksum_ok` False means the number cannot be a genuine Aadhaar."""
    out = []
    for ln in lines:
        toks = [w.text.strip() for w in ln]
        i = 0
        while i < len(ln):
            t = toks[i]
            if re.fullmatch(r"\d{12}", t):
                ws = [ln[i]]
                step = 1
            elif (i + 2 < len(ln) and all(re.fullmatch(r"\d{4}", toks[i + k]) for k in range(3))
                  and not (i + 3 < len(ln) and re.fullmatch(r"\d{4}", toks[i + 3]))      # 4th group => 16-digit VID
                  and not (i > 0 and re.fullmatch(r"\d{4}", toks[i - 1]))):
                ws = ln[i:i + 3]
                step = 3
            else:
                i += 1
                continue
            digits = "".join(w.text for w in ws)
            out.append({"raw": " ".join(w.text for w in ws), "digits": digits, "bbox": _union(ws), "conf": _conf(ws),
                        "checksum_ok": verhoeff_valid(digits) and digits[0] not in "01"})
            i += step
    return out


# ---------------------------------------------------------------- PAN (5 letters, 4 digits, 1 letter)
PAN_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Z0-9])")


def find_pan(lines):
    out = []
    for ln in lines:
        for w in ln:
            m = PAN_RE.search(w.text.strip().upper())
            if m:
                out.append({"raw": m.group(0), "bbox": w.bbox, "conf": w.conf})
    return out


# ---------------------------------------------------------------- name above the DOB line (unlabelled, bilingual cards)
_STOP = ("government", "india", "aadhaar", "authority", "unique", "identification", "male", "female", "dob", "birth", "year")


def name_above_dob(lines, max_back=3):
    """Aadhaar front: the holder's English name is the closest Latin-script line above the DOB line.
    Hindi lines come out of an English OCR as garbage, so candidates must be 2-4 alphabetic capitalised tokens."""
    dob_idx = next((i for i, ln in enumerate(lines)
                    if any(re.fullmatch(r"(?i)dob:?|birth", w.text.strip(" :/")) for w in ln)), None)
    if dob_idx is None:
        return None
    for k in range(1, max_back + 1):
        i = dob_idx - k
        if i < 0:
            break
        ln = lines[i]
        toks = [w for w in ln if re.fullmatch(r"[A-Za-z][A-Za-z.'-]{1,}", w.text.strip())]
        if (2 <= len(toks) <= 4 and len(toks) == len(ln) and all(w.text[0].isupper() for w in toks)
                and not any(s in " ".join(w.text.lower() for w in toks) for s in _STOP)):
            return {"raw": " ".join(w.text for w in toks), "bbox": _union(toks), "conf": _conf(toks)}
    return None


# ---------------------------------------------------------------- multi-line address up to the PIN code
PIN_RE = re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)")


def address_to_pin(lines, max_lines=8, col_slack=60):
    """Aadhaar back: text after an English 'Address' label, continuing on the following lines (same column) until a
    6-digit PIN code is seen. Returns {raw, bbox, conf} or None."""
    for i, ln in enumerate(lines):
        idx = next((j for j, w in enumerate(ln) if re.fullmatch(r"(?i)address:?", w.text.strip())), None)
        if idx is None:
            continue
        x0 = ln[idx].bbox[0]
        words = list(ln[idx + 1:])
        pieces = [w for w in words]
        found = any(PIN_RE.search(w.text) for w in words)
        j = i + 1
        while not found and j < len(lines) and j <= i + max_lines:
            # keep words of the next line that sit in the same text column (ignore e.g. a QR code on the right)
            col = [w for w in lines[j] if w.bbox[0] >= x0 - col_slack]
            col = _first_cell(col)
            pieces += col
            found = any(PIN_RE.search(w.text) for w in col)
            j += 1
        pieces = _clean_tokens(pieces)
        if pieces and found:
            return {"raw": " ".join(w.text for w in pieces), "bbox": _union(pieces), "conf": _conf(pieces)}
    return None


def _clean_tokens(ws, min_conf=0.6):
    """Drop OCR junk (low-confidence words, stray symbols, lone 1-2 letter fragments from Devanagari/QR bleed)."""
    out = []
    for w in ws:
        t = w.text.strip()
        if w.conf < min_conf or not re.search(r"[A-Za-z0-9]", t):
            continue
        if re.fullmatch(r"[A-Za-z]{1,2}", t) and not (t.isupper() and len(t) == 1) and t.lower() not in ("up", "mp", "hp", "uk", "ap", "wb", "tn", "dl", "mh", "co", "c/o", "no", "st", "rd", "of", "by"):
            continue
        out.append(w)
    return out


def _first_cell(ws, gap=80):
    """Leading run of words with no horizontal gap larger than `gap` px."""
    if not ws:
        return ws
    run = [ws[0]]
    for a, b in zip(ws, ws[1:]):
        if b.bbox[0] - a.bbox[2] > gap:
            break
        run.append(b)
    return run


# ---------------------------------------------------------------- classification cues
STRONG_CUES = {  # a single hit is strong evidence for the type
    "id_card": ["aadhaar", "uidai", "government of india", "unique identification", "permanent account number",
                "income tax department"],
}
