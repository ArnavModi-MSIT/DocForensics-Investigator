"""Per-field comparators for cross-document checks.

A single strict string match produced false alarms on perfectly consistent documents ("R. K. Sharma" vs
"Rahul Kumar Sharma", "Flat 12, Sector 21" vs "12 Sector-21, Noida 201301", "Acme Pvt Ltd" vs "ACME PRIVATE LIMITED").
Each field type therefore gets its own rule and returns one of:
  same          identical after normalisation
  compatible    consistent but written differently (initials, omitted middle name, word order, suffixes, masked digits)
  near          small difference: could be an OCR slip or a minor alteration (worth a reviewer's look)
  different     clearly different values
  unverifiable  not enough information to compare (single-word name, address without PIN and little text, ...)
Only `near` and `different` produce findings; `unverifiable` produces none but is counted for coverage reporting.
"""
import re
from dataclasses import dataclass
from rapidfuzz import fuzz

HONORIFICS = {"mr", "mrs", "ms", "miss", "shri", "shree", "smt", "sri", "dr", "kumari", "late", "sh", "master"}
CORP_SUFFIX = {"pvt", "private", "ltd", "limited", "llp", "inc", "incorporated", "corp", "corporation", "co", "company",
               "the", "and", "of"}
ADDR_STOP = {"flat", "no", "house", "h", "plot", "near", "opp", "opposite", "c", "o", "s", "d", "w", "the", "of", "and",
             "in", "dist", "district", "po", "ps", "vill", "village", "post", "tehsil", "india", "to", "at", "by", "back",
             "behind", "next", "floor", "fl"}
ADDR_ABBR = {"rd": "road", "st": "street", "sec": "sector", "blk": "block", "apt": "apartment", "bldg": "building",
             "ngr": "nagar", "col": "colony", "extn": "extension", "ext": "extension", "mkt": "market"}
PIN_RE = re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)")


@dataclass
class Cmp:
    verdict: str
    sim: int
    note: str = ""


def _ratio(a, b):
    return int(round(fuzz.ratio(a, b)))


def _tok_match(a, b):
    """Token equality tolerant to a one-character OCR slip on longer words."""
    return a == b or (len(a) >= 5 and len(b) >= 5 and fuzz.ratio(a, b) >= 85)


# ------------------------------------------------------------------ names
def name_tokens(s):
    t = re.findall(r"[a-z]+", re.sub(r"[.'`]", " ", str(s).lower()))
    return [w for w in t if w not in HONORIFICS]


def compare_names(a, b):
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb:
        return Cmp("unverifiable", 0, "empty name")
    if sorted(ta) == sorted(tb):
        return Cmp("same", 100)
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    pool, exact, init = list(long_), 0, 0
    for s in short:                                              # exact matches first, then initials
        hit = next((l for l in pool if l == s), None)
        if hit is not None:
            pool.remove(hit); exact += 1
    rest = [s for s in short if s not in long_ or short.count(s) > long_.count(s)]
    left = []
    for s in rest:
        hit = next((l for l in pool if (len(s) == 1 and l.startswith(s)) or (len(l) == 1 and s.startswith(l))), None)
        if hit is not None:
            pool.remove(hit); init += 1
        else:
            left.append(s)
    if not left:                                                 # every token of the shorter name is accounted for
        if len(short) == len(long_):
            return Cmp("compatible", 100, "initials or word order differ")
        if len(short) >= 2 and exact >= 1:
            return Cmp("compatible", 100, "middle name(s) omitted in one document")
        return Cmp("unverifiable", 0, "one name is a single word: cannot be compared reliably")
    sim = int(fuzz.token_sort_ratio(" ".join(ta), " ".join(tb)))
    fuzzy = len(ta) == len(tb) and all(any(_tok_match(x, y) for y in tb) for x in ta)
    return Cmp("near" if (sim >= 80 or fuzzy) else "different", sim)


# ------------------------------------------------------------------ addresses
def addr_tokens(s):
    pins = PIN_RE.findall(str(s))
    t = re.findall(r"[a-z0-9]+", str(s).lower())
    t = [ADDR_ABBR.get(w, w) for w in t if w not in pins]
    return [w for w in t if w not in ADDR_STOP and (len(w) > 1 or w.isdigit())], pins


def _containment(ta, tb, fuzzy=True):
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if not short:
        return 0.0
    return sum(1 for s in short if any((_tok_match(s, l) if fuzzy else s == l) for l in long_)) / len(short)


def compare_addresses(a, b):
    ta, pa = addr_tokens(a)
    tb, pb = addr_tokens(b)
    c = _containment(ta, tb)
    sim = int(round(100 * c))
    if pa and pb:
        if set(pa).isdisjoint(pb):
            return Cmp("different", sim, f"different PIN codes ({pa[-1]} vs {pb[-1]})")
        return Cmp("compatible" if c >= 0.5 else "different", sim,
                   "same PIN code" if c >= 0.5 else "same PIN code but the address text differs")
    if c >= 0.6 and min(len(ta), len(tb)) >= 2:
        return Cmp("compatible", sim, "no PIN on one side; address words overlap")
    if min(len(ta), len(tb)) >= 3 and c < 0.25:
        return Cmp("different", sim, "address words do not overlap")
    return Cmp("unverifiable", sim, "address too short or without PIN to compare")


# ------------------------------------------------------------------ employers
def employer_tokens(s):
    t = re.findall(r"[a-z0-9]+", str(s).lower().replace("&", " and "))
    return [w for w in t if w not in CORP_SUFFIX]


def compare_employers(a, b):
    ta, tb = employer_tokens(a), employer_tokens(b)
    if not ta or not tb:
        return Cmp("unverifiable", 0, "empty employer")
    if sorted(ta) == sorted(tb):
        return Cmp("same", 100)
    if _containment(ta, tb, fuzzy=False) == 1.0 and min(len(ta), len(tb)) >= 1 and any(len(w) >= 3 for w in (ta if len(ta) <= len(tb) else tb)):
        return Cmp("compatible", 100, "one document uses a shorter company name")
    sim = int(fuzz.token_sort_ratio(" ".join(ta), " ".join(tb)))
    return Cmp("near" if sim >= 85 else "different", sim)


# ------------------------------------------------------------------ exact-style identifiers
def compare_exact(a, b, masked_ok=False):
    sa, sb = str(a), str(b)
    if sa == sb:
        return Cmp("same", 100)
    if masked_ok and (re.search(r"[Xx*]", sa) or re.search(r"[Xx*]", sb)):
        da, db = re.sub(r"\D", "", sa), re.sub(r"\D", "", sb)
        if len(da) >= 3 and len(db) >= 3:
            n = min(len(da), len(db), 4)
            return Cmp("compatible" if da[-n:] == db[-n:] else "different", 0,
                       "masked number: only the last digits were compared")
    sim = _ratio(sa, sb)
    return Cmp("near" if sim >= 80 else "different", sim)


def compare(key, a, b):
    if key in ("name", "father_name"):
        return compare_names(a, b)
    if key == "address":
        return compare_addresses(a, b)
    if key == "employer":
        return compare_employers(a, b)
    return compare_exact(a, b, masked_ok=(key in ("account_no", "aadhaar_no")))
