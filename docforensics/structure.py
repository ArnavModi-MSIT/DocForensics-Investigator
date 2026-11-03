"""Document structure: tables and sections.

Tables
  * transactions (bank statement style): found by their header row (Date / Description / Withdrawal / Deposit / Balance ...,
    also two-line headers). Numeric columns are located from where the amounts actually sit (they are right aligned and
    usually do NOT line up with the header text). A row starts at a date or at an amount; lines without either continue the
    narration of the row above; a date printed once per day is inherited by the rows below it.
  * components (salary slip style): Earnings | Amount | Deductions | Amount side by side.
  Checks: running balance (previous balance + deposit - withdrawal = balance), balance carried across pages, and
  earnings / deductions components adding up to their stated totals.
Sections
  Headings are lines that are larger than body text, or short ALL-CAPS / colon-terminated lines set apart by extra space.
  Every field and table is attached to the section it sits in.
Everything here works on OCR word boxes only (no model), is read-only with respect to extracted fields, and reports what it
could not read instead of guessing.
"""
import os, re, statistics
import cv2
import numpy as np
from rapidfuzz import fuzz
from PIL import Image, ImageDraw
from .ocr import group_lines
from .extract import parse_date, parse_money, _best_label
from .schema import Finding, Field

# ----------------------------------------------------------------------------------------------- column vocabulary
KINDS = {
    "date": ["date", "txndate", "transactiondate", "postdate", "postingdate", "trandate", "bookingdate"],
    "value_date": ["valuedate", "valuedt", "valdate"],
    "desc": ["description", "narration", "particulars", "details", "transactiondetails", "remarks", "transactionremarks"],
    "ref": ["cheque", "chq", "chequeno", "chqno", "refno", "reference", "chqrefno", "instrumentno", "chequenumber", "refchqno"],
    "debit": ["withdrawal", "withdrawals", "debit", "debits", "dr", "withdrawalamt", "withdrawalamount", "debitamount", "paidout"],
    "credit": ["deposit", "deposits", "credit", "credits", "cr", "depositamt", "depositamount", "creditamount", "paidin"],
    "balance": ["balance", "closingbalance", "runningbalance", "availablebalance", "bal"],
    "amount": ["amount", "amt"],
    "earnings": ["earnings", "earning", "income"],
    "deductions": ["deductions", "deduction"],
}
NUMERIC = {"debit", "credit", "balance", "amount"}
MONEY_RE = re.compile(r"^\(?-?[\d,]*\d(?:\.\d{1,2})?\)?$")
DRCR = {"cr", "dr", "cr.", "dr."}
OPENING_RE = re.compile(r"balance\s*(forward|b/?f|brought)|opening\s*balance|b/?f\b", re.I)
REF_RE = re.compile(r"\b(?:UPI|IMPS|NEFT|RTGS)[/ -]*(\d{9,})", re.I)                  # payment references are unique per transaction
TOTAL_RE = re.compile(r"^\s*(total|gross|sub\s*total)", re.I)
NET_RE = re.compile(r"^\s*net\b", re.I)                          # "Net Pay" sits below the table, it is not a component


def _norm(t):
    return re.sub(r"[^a-z]", "", t.lower())


def col_kind(text):
    n = _norm(text)
    if not n:
        return None
    best, bs = None, 0
    for kind, names in KINDS.items():
        for nm in names:
            if n == nm:
                return kind
            if len(nm) >= 5 and len(n) >= 5:
                r = fuzz.ratio(n, nm)
                if r >= 85 and r > bs:
                    best, bs = kind, r
    return best


def _is_money(w):
    t = w.text.strip().rstrip("*")
    if not MONEY_RE.match(t):
        return False
    digits = sum(c.isdigit() for c in t)
    return digits >= 3 or "," in t or "." in t


def _money_value(text):
    t = text.strip()
    neg = t.startswith("(") and t.endswith(")") or t.startswith("-")
    v = parse_money(re.sub(r"[()\-]", "", t))
    return None if v is None else (-v if neg else v)


def _union(ws):
    return (min(w.bbox[0] for w in ws), min(w.bbox[1] for w in ws), max(w.bbox[2] for w in ws), max(w.bbox[3] for w in ws))


def _med(xs, default=0):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else default


def _lm(ln):
    hs = [w.bbox[3] - w.bbox[1] for w in ln]
    return {"h": _med(hs, 12), "cy": sum((w.bbox[1] + w.bbox[3]) / 2 for w in ln) / len(ln),
            "y0": min(w.bbox[1] for w in ln), "y1": max(w.bbox[3] for w in ln), "x0": min(w.bbox[0] for w in ln)}


def _cells(words, gap):
    cells = []
    for w in sorted(words, key=lambda w: w.bbox[0]):
        if cells and w.bbox[0] - cells[-1]["x1"] <= gap:
            cells[-1]["ws"].append(w)
            cells[-1]["x1"] = max(cells[-1]["x1"], w.bbox[2])
        else:
            cells.append({"ws": [w], "x0": w.bbox[0], "x1": w.bbox[2]})
    def fill(c):
        h = _med([w.bbox[3] - w.bbox[1] for w in c["ws"]], 12)
        c["text"] = " ".join(w.text for w in sorted(c["ws"], key=lambda w: (round(w.bbox[1] / max(8, 0.7 * h)), w.bbox[0])))
        c["bbox"] = _union(c["ws"])
        c["kind"] = col_kind(c["text"])
    out = []
    for c in cells:
        fill(c)
        if c["kind"] is None and len(c["ws"]) > 1:             # 'Value Description': two column titles closer than the gap limit; a title word starts its own cell
            parts, cur = [], []
            for w in sorted(c["ws"], key=lambda w: w.bbox[0]):
                if cur and col_kind(w.text):
                    parts.append(cur)
                    cur = []
                cur.append(w)
            parts.append(cur)
            if len(parts) > 1:
                for ws in parts:
                    sub = {"ws": ws, "x0": min(w.bbox[0] for w in ws), "x1": max(w.bbox[2] for w in ws)}
                    fill(sub)
                    out.append(sub)
                continue
        out.append(c)
    return out


# ----------------------------------------------------------------------------------------------- header detection
def _find_headers(lines):
    metrics = [_lm(ln) for ln in lines]
    cands = []
    for i, m in enumerate(metrics):
        band = [w for j, ln in enumerate(lines) if abs(metrics[j]["cy"] - m["cy"]) <= 1.3 * m["h"] + 3 for w in ln]
        cells = _cells(band, gap=max(30, 2.3 * m["h"]))
        kinds = [c["kind"] for c in cells if c["kind"]]
        distinct = set(kinds)
        tx = len(distinct) >= 3 and (distinct & NUMERIC) and (distinct & {"date", "desc", "value_date"})
        comp = {"earnings", "deductions"} <= distinct and "amount" in distinct
        if tx or comp:
            cands.append({"score": len(distinct), "i": i, "kind": "components" if comp and not tx else "transactions",
                          "cells": cells, "band": band, "y": m["y0"], "bottom": max(w.bbox[3] for w in band), "h": m["h"]})
    accepted, taken = [], set()
    for c in sorted(cands, key=lambda c: (-c["score"], c["y"])):
        ids = {id(w) for w in c["band"]}
        if ids & taken:
            continue
        taken |= ids
        accepted.append(c)
    return sorted(accepted, key=lambda c: c["y"]), metrics


# ----------------------------------------------------------------------------------------------- numeric column anchors
def _anchors(num_cols, money_words, h):
    """x position of each numeric column's right edge, learned from the amounts themselves."""
    pref = [w for w in money_words if re.search(r"\.\d{2}$", w.text.strip())] or money_words
    edges = sorted(w.bbox[2] for w in pref)
    tol = max(24, 2 * h)
    clusters = []
    for e in edges:
        if clusters and e - clusters[-1][-1] <= tol:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    centers = [sum(c) / len(c) for c in clusters]
    if not centers:
        return [c["x1"] for c in num_cols]
    delta = centers[-1] - num_cols[-1]["x1"]                       # the last numeric column (balance) is nearly always filled
    pred = [c["x1"] + delta for c in num_cols]
    anchors = list(pred)
    for k in range(len(num_cols)):
        mine = [ctr for ctr in centers if min(range(len(pred)), key=lambda j: abs(ctr - pred[j])) == k]
        if mine:
            anchors[k] = sum(mine) / len(mine)
    return anchors


# ----------------------------------------------------------------------------------------------- transactions
def _build_transactions(hdr, body, page_w):
    cells = hdr["cells"]
    h = hdr["h"]
    num_cols = [c for c in cells if c["kind"] in NUMERIC]
    text_cols = [c for c in cells if c["kind"] not in NUMERIC]
    if not num_cols or not text_cols:
        return None
    xmin = num_cols[0]["x0"] - 0.06 * page_w
    money = [w for ln in body for w in ln if _is_money(w) and w.bbox[0] >= xmin]
    anchors = _anchors(num_cols, money, h)
    tol_assign = max(40, 3 * h)
    desc = next((c for c in text_cols if c["kind"] == "desc"), text_cols[0])
    date_col = next((c for c in text_cols if c["kind"] == "date"), None)
    margin = max(25, 1.8 * h)
    rows, cur, prev_y1 = [], None, hdr["bottom"]
    pitch, last_cy = [], None                                       # distances between row starts: airy tables have rows far apart
    for ln in body:
        m = _lm(ln)
        lim = 4.5 * max(h, m["h"])
        lim = max(lim, 9 * max(h, m["h"])) if not pitch else max(lim, 2.2 * statistics.median(pitch))        # the first two rows set the pitch
        if m["y0"] - prev_y1 > lim:                                 # large vertical gap: the table is over (footer, next block)
            break
        per, last_num = {}, None                                    # (type, column index) -> words
        ln_sorted = sorted(ln, key=lambda w: w.bbox[0])
        for w in ln_sorted:
            if w.text.lower() in DRCR and last_num is not None:        # '1,234.00 Cr': suffix belongs to the amount
                per[("n", last_num)].append(w)
                continue
            if _is_money(w) and w.bbox[0] >= xmin:
                k = min(range(len(num_cols)), key=lambda j: abs(w.bbox[2] - anchors[j]))
                if abs(w.bbox[2] - anchors[k]) <= tol_assign:
                    per.setdefault(("n", k), []).append(w)
                    last_num = k
                    continue
            last_num = None
            cands = [j for j, c in enumerate(text_cols) if c["x0"] - margin <= w.bbox[0]]
            j = max(cands) if cands else 0
            per.setdefault(("t", j), []).append(w)
        vals = {}
        for (typ, idx), ws in per.items():
            col = num_cols[idx] if typ == "n" else text_cols[idx]
            kind = col["kind"] or "other"
            vals.setdefault(kind, []).extend(ws)
        date_txt = " ".join(w.text for w in sorted(vals.get("date", []), key=lambda w: w.bbox[0]))
        iso = parse_date(date_txt) if date_txt else None
        has_amt = any(k in vals for k in NUMERIC)
        first_x = ln_sorted[0].bbox[0]
        if iso or has_amt:
            if last_cy is not None:
                pitch.append(m["cy"] - last_cy)
            last_cy = m["cy"]
            cur = {"n": len(rows) + 1, "date": iso, "inherited_date": iso is None, "cells": {}, "words": []}
            if iso is None and rows:
                cur["date"] = rows[-1]["date"]
            rows.append(cur)
        elif cur is not None and first_x >= desc["x0"] - margin:    # narration continues
            pass
        else:
            break                                                    # text starting left of the narration column: footer
        for kind, ws in vals.items():
            c = cur["cells"].setdefault(kind, {"words": []})
            c["words"].extend(ws)
        cur["words"].extend(ln)
        prev_y1 = m["y1"]
    out = []
    for r in rows:
        cells_out, vals = {}, {}
        for kind, c in r["cells"].items():
            ws = sorted(c["words"], key=lambda w: (round(w.bbox[1] / 8), w.bbox[0]))
            text = " ".join(w.text for w in ws)
            cells_out[kind] = {"text": text, "bbox": list(_union(ws)), "conf": round(sum(w.conf for w in ws) / len(ws), 3)}
            if kind in NUMERIC:
                sign = -1 if any(w.text.lower() in ("dr", "dr.") for w in ws) and kind == "balance" else 1
                nums = [w for w in ws if w.text.lower() not in DRCR]
                v = _money_value(nums[0].text) if nums else None
                vals[kind] = None if v is None else sign * v
        desc_txt = cells_out.get("desc", {}).get("text", "")
        role = ("opening" if OPENING_RE.search(desc_txt) else "closing" if re.search(r"closing\s*balance", desc_txt, re.I)
                else "total" if re.match(r"\s*(grand\s+)?total", desc_txt, re.I) else "txn")
        out.append({"n": r["n"], "role": role, "date": r["date"], "inherited_date": r["inherited_date"], "desc": desc_txt,
                    "debit": vals.get("debit"), "credit": vals.get("credit"), "balance": vals.get("balance"),
                    "amount": vals.get("amount"), "cells": cells_out, "bbox": list(_union(r["words"])),
                    "conf": round(min([c["conf"] for k, c in cells_out.items() if k in NUMERIC] or [1.0]), 3)})
    keep = [r for r in out if any(r[k] is not None for k in ("debit", "credit", "balance", "amount"))]
    if len(keep) < 2:
        return None
    bbox = [min(hdr["cells"][0]["x0"], min(r["bbox"][0] for r in out)), hdr["y"],
            max(max(c["x1"] for c in cells), max(r["bbox"][2] for r in out)), max(r["bbox"][3] for r in out)]
    return {"kind": "transactions", "bbox": bbox, "rows": out,
            "header": [{"kind": c["kind"] or "other", "text": c["text"], "bbox": list(c["bbox"])} for c in cells],
            "columns": [{"kind": c["kind"] or "other", "x0": c["x0"], "x1": c["x1"]} for c in cells],
            "anchors": {c["kind"]: round(a) for c, a in zip(num_cols, anchors)}}


def check_transactions(table, start_balance=None):
    """Running balance: previous balance + deposit - withdrawal must equal the balance printed on the row."""
    prev, breaks, checked, unread = start_balance, [], 0, 0
    for r in table["rows"]:
        b = r["balance"]
        if r["role"] == "opening":
            prev = b if b is not None else prev
            continue
        if r["role"] in ("closing", "total"):
            continue
        if r["debit"] is None and r["credit"] is None:
            unread += 1
            prev = b if b is not None else None
            continue
        if b is None:
            prev = None
            unread += 1
            continue
        if prev is not None:
            exp = prev + (r["credit"] or 0) - (r["debit"] or 0)
            checked += 1
            if abs(exp - b) > max(0.01, 1e-6 * abs(b)):
                breaks.append({"row": r["n"], "date": r["date"], "desc": r["desc"][:50], "previous": prev, "debit": r["debit"],
                               "credit": r["credit"], "expected": round(exp, 2), "printed": b, "difference": round(b - exp, 2),
                               "conf": r["conf"], "bbox": r["cells"].get("balance", {}).get("bbox", r["bbox"])})
        prev = b
    return {"rows": len(table["rows"]), "checked": checked, "unreadable_rows": unread, "breaks": breaks}


# ----------------------------------------------------------------------------------------------- components (earnings | deductions)
def _build_components(hdr, body, page_w):
    cells = hdr["cells"]
    h = hdr["h"]
    sides = []                                                      # (side name, text col, amount col) pairs, left to right
    for i, c in enumerate(cells):
        if c["kind"] in ("earnings", "deductions"):
            amt = next((a for a in cells[i + 1:] if a["kind"] == "amount"), None)
            if amt:
                sides.append((c["kind"], c, amt))
    if not sides:
        return None
    rows, prev_y1 = [], hdr["bottom"]
    pitch, last_cy = [], None
    for ln in body:
        m = _lm(ln)
        lim = 4.5 * max(h, m["h"])
        lim = max(lim, 9 * max(h, m["h"])) if not pitch else max(lim, 2.2 * statistics.median(pitch))        # the first two rows set the pitch
        if m["y0"] - prev_y1 > lim:
            break
        ln_sorted = sorted(ln, key=lambda w: w.bbox[0])
        found = False
        for si, (side, tcol, acol) in enumerate(sides):
            lo = tcol["x0"] - 1.5 * h
            hi = sides[si + 1][1]["x0"] - 1.5 * h if si + 1 < len(sides) else 1e9
            ws = [w for w in ln_sorted if lo <= w.bbox[0] < hi]
            nums = [w for w in ws if _is_money(w) and w.bbox[2] >= acol["x0"] - 2.5 * h]
            txt = [w for w in ws if w not in nums]
            if not nums or not txt:
                continue
            v = _money_value(nums[-1].text)
            if v is None:
                continue
            label = " ".join(w.text for w in txt)
            if NET_RE.match(label):
                found = False
                break
            if last_cy is None or abs(m["cy"] - last_cy) > 0.5 * m["h"]:
                if last_cy is not None:
                    pitch.append(m["cy"] - last_cy)
                last_cy = m["cy"]
            rows.append({"n": len(rows) + 1, "side": side, "label": label, "value": v,
                         "role": "total" if TOTAL_RE.match(label) else "item", "bbox": list(_union(ws)),
                         "conf": round(sum(w.conf for w in ws) / len(ws), 3)})
            found = True
        if not found:
            break
        prev_y1 = m["y1"]
    if len(rows) < 2:
        return None
    return {"kind": "components", "rows": rows, "header": [{"kind": c["kind"] or "other", "text": c["text"], "bbox": list(c["bbox"])} for c in cells],
            "columns": [{"kind": c["kind"] or "other", "x0": c["x0"], "x1": c["x1"]} for c in cells],
            "bbox": [min(c["x0"] for c in cells), hdr["y"], max(c["x1"] for c in cells), max(r["bbox"][3] for r in rows)]}


def check_components(table, doc=None):
    out = []
    for side in ("earnings", "deductions"):
        items = [r for r in table["rows"] if r["side"] == side and r["role"] == "item"]
        total = next((r for r in table["rows"] if r["side"] == side and r["role"] == "total"), None)
        if not items:
            continue
        s = round(sum(r["value"] for r in items), 2)
        stated, src, bbox = None, None, None
        if total:
            stated, src, bbox = total["value"], f"'{total['label']}' row", total["bbox"]
        elif doc is not None:
            f = doc.get("gross" if side == "earnings" else "deductions")
            if f and isinstance(f.value, float):
                stated, src, bbox = f.value, f"the {f.key} field", list(f.bbox)
        if stated is not None and abs(s - stated) > max(0.01, 1e-6 * abs(stated)):
            out.append({"side": side, "sum": s, "stated": stated, "source": src, "difference": round(stated - s, 2),
                        "bbox": bbox, "conf": min(r["conf"] for r in items)})
    return out


# ----------------------------------------------------------------------------------------------- sections
SECTION_KINDS = [("personal_details", ("personal", "applicant", "identity", "employee", "holder", "customer", "details of")),
                 ("address", ("address", "residen")), ("earnings", ("earning", "income")), ("deductions", ("deduction",)),
                 ("summary", ("summary", "overview")), ("transactions", ("transaction",)),
                 ("declaration", ("declaration", "undertaking", "signature", "authoris", "consent")),
                 ("employment", ("employment", "employer", "occupation")), ("account_details", ("account", "bank details", "branch"))]


def _section_kind(title):
    low = title.lower()
    if "earning" in low and "deduction" in low:
        return "earnings_deductions"
    for kind, keys in SECTION_KINDS:
        if any(k in low for k in keys):
            return kind
    return "other"


def detect_sections(doc, tables):
    lines = group_lines(doc.words)
    if not lines:
        return []
    metrics = [_lm(ln) for ln in lines]
    body_h = _med([m["h"] for m in metrics], 12)
    gaps = [metrics[i + 1]["y0"] - metrics[i]["y1"] for i in range(len(lines) - 1) if metrics[i + 1]["y0"] > metrics[i]["y1"]]
    pitch = max(_med(gaps, 12), 6)                                   # typical blank space between two body lines
    in_table = lambda m: any(t["bbox"][1] - 4 <= m["cy"] <= t["bbox"][3] + 4 for t in tables)
    heads, bigs = [], set()
    for i, (ln, m) in enumerate(zip(lines, metrics)):
        text = " ".join(w.text for w in ln)
        letters = sum(c.isalpha() for c in text)
        if in_table(m) or letters < 4 or len(ln) > 8 or sum(c.isdigit() for c in text) > 0.3 * max(1, len(text)):
            continue
        if ":" in text.strip(" :") and not text.strip().endswith(":"):             # label: value line
            continue
        key, _sc = _best_label(" ".join(w.text for w in ln[:3]))
        if key and len(ln) > len(re.findall(r"\S+", " ".join(w.text for w in ln[:3]))):
            continue
        big = m["h"] >= 1.25 * body_h
        if big:
            bigs.add(i)
        caps = text.upper() == text and letters >= 4
        colon = text.strip().endswith(":")
        gap_before = i == 0 or (m["y0"] - metrics[i - 1]["y1"]) >= 1.1 * pitch
        if big or ((caps or colon) and gap_before):
            heads.append(i)
    merged = []                                                                   # adjacent heading lines form one heading
    for i in heads:
        prev = merged[-1][-1] if merged else None
        if (prev is not None and i == prev + 1 and metrics[i]["y0"] - metrics[prev]["y1"] < 1.5 * pitch
                and 0.8 <= metrics[i]["h"] / max(1, metrics[prev]["h"]) <= 1.25):          # same-size neighbours form one heading
            merged[-1].append(i)
        else:
            merged.append([i])
    spans = []
    first = merged[0][0] if merged else len(lines)
    if first > 0:
        spans.append((None, 0, first))
    for k, hs in enumerate(merged):
        end = merged[k + 1][0] if k + 1 < len(merged) else len(lines)
        spans.append((hs, hs[0], end))
    sections = []
    for hs, a, b in spans:
        ls = lines[a:b]
        words = [w for ln in ls for w in ln]
        bbox = list(_union(words))
        title = " ".join(w.text for i in hs for w in lines[i]).strip(" :") if hs else "(top of page)"
        inside = lambda f: any(t["bbox"][1] - 2 <= (f.bbox[1] + f.bbox[3]) / 2 <= t["bbox"][3] + 2 for t in tables)
        fields = [f.key for f in doc.fields if bbox[1] - 2 <= (f.bbox[1] + f.bbox[3]) / 2 <= bbox[3] + 2 and not inside(f)]
        tids = [t["tid"] for t in tables if bbox[1] - 2 <= (t["bbox"][1] + t["bbox"][3]) / 2 <= bbox[3] + 2]
        kind = "title" if (hs and hs[0] in bigs and hs[0] <= 1 and not sections) else _section_kind(title)
        if tids and kind == "other":
            kind = "transactions" if any(t["kind"] == "transactions" for t in tables if t["tid"] in tids) else "earnings_deductions"
        sections.append({"sid": f"S{len(sections) + 1}", "title": title[:80], "kind": kind, "bbox": bbox, "n_lines": len(ls),
                         "fields": fields, "tables": tids, "has_heading": bool(hs)})
    return sections


# ----------------------------------------------------------------------------------------------- layout and placement
def detect_layout(doc):
    """One-line layout label from what the page actually contains, with the evidence, plus the information hierarchy counts."""
    lab = [f for f in doc.fields if f.match in ("fuzzy", "fuzzy-cell", "semantic")]
    pat = [f for f in doc.fields if f.match == "pattern"]
    tx = [t for t in doc.tables if t["kind"] == "transactions"]
    comp = [t for t in doc.tables if t["kind"] == "components"]
    heads = [s for s in doc.sections if s["has_heading"]]
    ev = [f"{len(lab)} labelled field(s)", f"{len(pat)} label-free field(s)", f"{len(heads)} heading(s)"]
    if tx:
        label = "statement: label-value header block + transaction table" if lab else "transaction table"
        ev.append(f"transaction table with {len(tx[0]['rows'])} rows")
    elif comp:
        label = "slip: label-value block + earnings/deductions table" if lab else "earnings/deductions table"
        ev.append(f"components table with {len(comp[0]['rows'])} rows")
    elif doc.doc_type == "id_card" and len(pat) >= len(lab):
        label = "label-free ID card (stacked values, pattern-based)"
    elif len(lab) >= 3:
        label = "label-value form" + (f" in {len(heads)} sections" if len(heads) >= 2 else "")
    elif doc.fields:
        label = "sparse form (few labelled fields)"
    else:
        label = "unstructured text"
    cells = sum(len(r.get("cells", {})) for t in doc.tables for r in t["rows"] if t["kind"] == "transactions")
    return {"label": label, "evidence": ev,
            "hierarchy": {"sections": len(doc.sections), "headed_sections": len(heads), "fields": len(doc.fields),
                          "tables": len(doc.tables), "table_rows": sum(len(t["rows"]) for t in doc.tables), "table_cells": cells}}


IDENTITY_KEYS = {"name", "father_name", "dob", "address", "id_number", "aadhaar_no", "epic_no"}
AMOUNT_KEYS = {"gross", "net_pay", "deductions", "salary_credit", "declared_income"}
FINANCE_SECTIONS = {"earnings", "deductions", "earnings_deductions", "transactions"}


def misplaced_in_sections(doc):
    """Identity fields found inside a financial section, or amounts inside an address section (fields inside tables are excluded)."""
    out = []
    for s in doc.sections:
        for key in s["fields"]:
            bad = (key in IDENTITY_KEYS and s["kind"] in FINANCE_SECTIONS) or (key in AMOUNT_KEYS and s["kind"] == "address")
            f = doc.get(key)
            if bad and f:
                out.append({"key": key, "section": s["title"], "kind": s["kind"], "field": f})
    return out


# ----------------------------------------------------------------------------------------------- driver
def detect_structure(doc, audit=None):
    """Fills doc.tables and doc.sections (recognised pages only; a failure here never aborts the pipeline)."""
    doc.tables, doc.sections = [], []
    try:
        lines = group_lines(doc.words)
        headers, metrics = _find_headers(lines)
        for n, hdr in enumerate(headers):
            band_ids = {id(w) for w in hdr["band"]}
            nxt = headers[n + 1]["y"] if n + 1 < len(headers) else 1e9
            body = [ln for ln, m in zip(lines, metrics) if m["y0"] >= hdr["bottom"] - 2 and m["y0"] < nxt - 2
                    and not any(id(w) in band_ids for w in ln)]
            t = (_build_transactions(hdr, body, doc.width) if hdr["kind"] == "transactions"
                 else _build_components(hdr, body, doc.width))
            if t:
                t["tid"] = f"T{len(doc.tables) + 1}"
                t["checks"] = (check_transactions(t) if t["kind"] == "transactions" else {"mismatches": check_components(t, doc)})
                doc.tables.append(t)
        doc.sections = detect_sections(doc, doc.tables)
        doc.facts["layout"] = detect_layout(doc)
    except Exception as e:                                                      # structure is additive: never break extraction
        if audit:
            audit.log("structure_failed", doc_id=doc.doc_id, error=f"{type(e).__name__}: {str(e)[:80]}")
        return doc
    if audit:
        audit.log("structure", doc_id=doc.doc_id, sections=[(s["title"], s["kind"]) for s in doc.sections][:12],
                  tables=[(t["tid"], t["kind"], len(t["rows"])) for t in doc.tables])
    return doc


# ----------------------------------------------------------------------------------------------- fields derived from tables
PAYROLL_RE = re.compile(r"(?i)\b(payroll|salary|wages?|stipend|remuneration|pay\s*credit|sal\s*cr)\b")
PAYER_RE = re.compile(r"(?i)^\W*(payroll|salary|wages?|stipend|remuneration|pay\s*credit|sal\s*cr)\W*(credit(ed)?|payment|deposit)?\W*(from|by|-|:|\|)?\s*")


def derive_table_fields(doc, audit=None):
    """Bank statements: the salary arrives as a row of the transaction table, not as a labelled field. Payroll-like credits
    (description says payroll / salary / wages, credit column positive) are kept in doc.facts['payroll_credits'] with date,
    amount, payer and the cell's box; the latest one becomes the `salary_credit` field when none was labelled. The opening
    balance row fills `opening_balance` the same way."""
    if doc.doc_type != "bank_statement":
        return
    credits = []
    for t in doc.tables:
        if t["kind"] != "transactions":
            continue
        for r in t["rows"]:
            if r["role"] == "opening" and r["balance"] is not None and not doc.get("opening_balance"):
                bb = r["cells"].get("balance", {})
                doc.fields.append(Field("opening_balance", "opening balance row", bb.get("text", str(r["balance"])), float(r["balance"]),
                                        tuple(bb.get("bbox") or r["bbox"]), bb.get("conf", r["conf"]), "table", 0.8))
            if r["role"] == "txn" and r["credit"] and r["credit"] > 0 and PAYROLL_RE.search(r["desc"] or ""):
                cb = r["cells"].get("credit", {})
                payer = PAYER_RE.sub("", r["desc"] or "").strip(" -:|")
                credits.append({"amount": float(r["credit"]), "date": r["date"], "desc": r["desc"], "payer": payer or None,
                                "bbox": list(cb.get("bbox") or r["bbox"]), "conf": cb.get("conf", r["conf"]), "raw": cb.get("text", "")})
    doc.facts["payroll_credits"] = credits
    if credits and not doc.get("salary_credit"):
        c = max(credits, key=lambda c: c["date"] or "")
        doc.fields.append(Field("salary_credit", "payroll credit (transaction table)", c["raw"] or f"{c['amount']:,.0f}", c["amount"],
                                tuple(c["bbox"]), c["conf"], "table", 0.8))
    if audit and credits:
        audit.log("payroll_credits", doc_id=doc.doc_id, credits=[(c["date"], c["amount"], c["payer"]) for c in credits][:6])


# ----------------------------------------------------------------------------------------------- duplicate pages
DUP_PAGE_MIN = 0.95          # calibrated: different pages (even same layout) <= 0.78; same page re-saved / shifted 2 px >= 0.99


def _ink_map(path, w=320):
    try:
        g = cv2.imread(path, 0)
        g = cv2.resize(g, (w, int(g.shape[0] * w / g.shape[1])), interpolation=cv2.INTER_AREA)
    except Exception:
        return None
    ink = g < np.median(g) * 0.72
    return ink if ink.sum() >= 150 else None                                     # a blank page has nothing to compare


def _ink_overlap(a, b):
    """Share of printed pixels of each page that is also printed (within 1 px) on the other; the smaller of the two."""
    if a is None or b is None or abs(a.shape[0] - b.shape[0]) > 0.05 * a.shape[0]:
        return None
    h = min(a.shape[0], b.shape[0])
    a, b = a[:h], b[:h]
    k = np.ones((3, 3), np.uint8)
    da, db = cv2.dilate(a.astype(np.uint8), k).astype(bool), cv2.dilate(b.astype(np.uint8), k).astype(bool)
    return float(min((a & db).sum() / a.sum(), (b & da).sum() / b.sum()))


# ----------------------------------------------------------------------------------------------- findings
def _tref(doc, label, value, bbox):
    return {"doc_id": doc.doc_id, "field": label, "value": value, "bbox": list(bbox) if bbox else None}


def table_findings(docs):
    out = []
    n = [0]

    def nid():
        n[0] += 1
        return f"T-{n[0]:03d}"
    by_file = {}
    for d in docs:
        by_file.setdefault(d.path, []).append(d)
    for d in docs:                                                              # values sitting in the wrong kind of section
        for m in misplaced_in_sections(d):
            f = m["field"]
            out.append(Finding(nid(), "intra", f"'{m['key']}' found inside the '{m['section']}' section of {d.doc_id}", "low", 0.5, "observed",
                               f"The {m['key'].replace('_', ' ')} value {f.raw!r} sits in a section titled {m['section']!r} ({m['kind'].replace('_', ' ')}), "
                               "where this kind of information is not expected. It may be misplaced, or the heading may belong to a neighbouring block. Verify manually.",
                               [_tref(d, m["key"], f.raw, f.bbox)], method="section placement check"))
    for d in docs:
        for t in d.tables:
            chk = t["checks"]
            if t["kind"] == "transactions":
                br = chk["breaks"]
                if br:
                    weak = min(b["conf"] for b in br) < 0.8
                    lines = "; ".join(
                        f"row {b['row']}{' (' + b['date'] + ')' if b['date'] else ''}: previous balance {b['previous']:,.2f}"
                        f"{' - debit ' + format(b['debit'], ',.2f') if b['debit'] else ''}"
                        f"{' + credit ' + format(b['credit'], ',.2f') if b['credit'] else ''} = {b['expected']:,.2f}, but the statement prints "
                        f"{b['printed']:,.2f} (difference {b['difference']:+,.2f})" for b in br[:4])
                    out.append(Finding(nid(), "intra", f"Running balance does not reconcile in {d.doc_id} ({len(br)} of {chk['checked']} rows)",
                                       "medium" if weak else "high", 0.7 if weak else 0.85, "inconsistency",
                                       f"Each row's balance must equal the previous balance plus deposits minus withdrawals. {lines}"
                                       + (f" (+{len(br) - 4} more)" if len(br) > 4 else "")
                                       + (" OCR confidence on some of these cells is below 0.80, so a misread digit is possible; verify against the original."
                                          if weak else " A single altered amount or balance breaks this chain."),
                                       [_tref(d, f"{t['tid']} row {b['row']} balance", b["printed"], b["bbox"]) for b in br[:6]],
                                       metrics={"breaks": len(br), "checked": chk["checked"]}, method="table running-balance check"))
                if chk["rows"] and chk["unreadable_rows"] > 0.25 * chk["rows"]:
                    out.append(Finding(nid(), "missing", f"Part of the transaction table in {d.doc_id} could not be read", "low", 0.8, "observed",
                                       f"{chk['unreadable_rows']} of {chk['rows']} table rows have no readable amount or balance, so the running-balance "
                                       f"check covered {chk['checked']} rows only.", [_tref(d, t["tid"], None, t["bbox"])],
                                       method="table coverage"))
            else:
                for m in chk["mismatches"]:
                    out.append(Finding(nid(), "intra", f"{m['side'].capitalize()} components do not add up to the stated total in {d.doc_id}",
                                       "high" if m["conf"] >= 0.8 else "medium", 0.8, "inconsistency",
                                       f"The {m['side']} items add up to {m['sum']:,.2f}, but {m['source']} states {m['stated']:,.2f} "
                                       f"(difference {m['difference']:+,.2f}). One component or the total may have been altered or misread.",
                                       [_tref(d, f"{t['tid']} {m['side']} total", m["stated"], m["bbox"])], method="table component-sum check"))
    for path, ds in by_file.items():                                            # duplicated rows / references
        groups = {}
        for d in ds:
            for t in d.tables:
                if t["kind"] != "transactions":
                    continue
                for r in t["rows"]:
                    if r["role"] != "txn":
                        continue
                    moved = (r["debit"] or 0) or (r["credit"] or 0)
                    if r["balance"] is not None and moved:
                        groups.setdefault(("row", r["date"], re.sub(r"[^a-z0-9]", "", r["desc"].lower())[:30], r["debit"], r["credit"], r["balance"]), []).append((d, t, r))
                    for ref in set(REF_RE.findall(r["desc"])):
                        groups.setdefault(("ref", ref, "d" if r["debit"] else "c", moved), []).append((d, t, r))
        flagged = set()
        for key, items in groups.items():
            if len(items) < 2 or (key[0] == "ref" and frozenset(id(r) for _, _, r in items) in flagged):
                continue
            flagged.add(frozenset(id(r) for _, _, r in items))
            rows = ", ".join(f"{d.doc_id} row {r['n']}" for d, t, r in items[:4])
            if key[0] == "row":
                out.append(Finding(nid(), "intra", f"Identical transaction row appears {len(items)} times ({os.path.basename(path)})", "medium", 0.75, "inconsistency",
                                   f"{rows} have the same date, description, amounts and printed balance ({items[0][2]['desc'][:40]!r}). After a non-zero movement a genuine "
                                   "running balance cannot repeat, so a row was probably copied or printed twice.",
                                   [_tref(d, f"{t['tid']} row {r['n']}", r["balance"], r["bbox"]) for d, t, r in items[:4]],
                                   method="duplicate table row check"))
            else:
                out.append(Finding(nid(), "intra", f"Same transaction reference appears on {len(items)} rows ({os.path.basename(path)})", "medium", 0.65, "inconsistency",
                                   f"Reference {key[1]} is used by {rows}, in the same direction and for the same amount. Payment references are unique per transaction "
                                   "(a reversal would be in the opposite direction), so this may be a duplicated entry.",
                                   [_tref(d, f"{t['tid']} row {r['n']}", key[1], r["bbox"]) for d, t, r in items[:4]],
                                   method="duplicate reference check"))
    rec = [d for d in docs if d.doc_type != "unknown"]                           # near-identical pages submitted twice
    inks = [(d, _ink_map(d.image_path)) for d in rec]
    for i in range(len(inks)):
        for j in range(i + 1, len(inks)):
            sc = _ink_overlap(inks[i][1], inks[j][1])
            if sc is not None and sc >= DUP_PAGE_MIN:
                a, b = inks[i][0], inks[j][0]
                out.append(Finding(nid(), "intra", f"Pages {a.doc_id} and {b.doc_id} are near-identical", "low", 0.7, "observed",
                                   f"The printed content of the two pages overlaps {sc * 100:.0f}% (different pages score below 80%, even with the same layout): "
                                   "the same document may have been submitted twice, or a page was copied. A copy with one value changed also scores this high.",
                                   [_tref(a, None, None, None), _tref(b, None, None, None)], metrics={"overlap": sc}, method="duplicate page check"))
    for path, ds in by_file.items():                                            # printed TOTAL row vs the sum of all rows
        txs = [(d, t) for d in ds for t in d.tables if t["kind"] == "transactions"]
        tot = next(((d, r) for d, t in reversed(txs) for r in t["rows"] if r["role"] == "total"), None)
        if tot and len(txs) >= 1:
            sums = {k: round(sum((r[k] or 0) for d, t in txs for r in t["rows"] if r["role"] == "txn"), 2) for k in ("debit", "credit")}
            unread = sum(t["checks"]["unreadable_rows"] for d, t in txs)
            for k, label in (("debit", "withdrawals"), ("credit", "deposits")):
                printed = tot[1][k]
                if printed is not None and abs(sums[k] - printed) > 0.01:
                    out.append(Finding(nid(), "intra", f"Transaction {label} do not add up to the printed total ({os.path.basename(path)})",
                                       "medium", 0.7, "inconsistency",
                                       f"The {len(txs)} page(s) of the statement table add up to {sums[k]:,.2f} in {label}, but the TOTAL row prints {printed:,.2f} "
                                       f"(difference {printed - sums[k]:+,.2f}). A row may have been removed or altered"
                                       + (f", or {unread} row(s) could not be read." if unread else "; every row was readable."),
                                       [_tref(tot[0], f"total {label}", printed, tot[1]["cells"].get(k, {}).get("bbox"))],
                                       metrics={"sum": sums[k], "printed": printed}, method="table totals check"))
    for path, ds in by_file.items():                                            # balance carried from page to page
        prev_last = None
        for d in ds:
            tx = [t for t in d.tables if t["kind"] == "transactions"]
            if not tx:
                prev_last = None
                continue
            rows = tx[0]["rows"]
            if prev_last is not None and rows and rows[0]["role"] == "opening" and rows[0]["balance"] is not None:
                a = prev_last
                if abs(a[1] - rows[0]["balance"]) > 0.01:
                    out.append(Finding(nid(), "intra", f"Balance carried forward does not match the previous page ({d.doc_id})", "high", 0.8, "inconsistency",
                                       f"{a[0]} ends with balance {a[1]:,.2f}, but {d.doc_id} starts from {rows[0]['balance']:,.2f}.",
                                       [_tref(d, f"{tx[0]['tid']} balance forward", rows[0]["balance"], rows[0]["cells"].get("balance", {}).get("bbox"))],
                                       method="table page-continuity check"))
            last = next((r for r in reversed(rows) if r["balance"] is not None and r["role"] != "closing"), None)
            prev_last = (d.doc_id, last["balance"]) if last else None
    return out


# ----------------------------------------------------------------------------------------------- export / drawing
def structure_dict(doc):
    """JSON-friendly view for the dossier (rows keep values, confidence and boxes)."""
    return {"layout": doc.facts.get("layout"), "sections": doc.sections,
            "tables": [{k: v for k, v in t.items() if k != "anchors"} for t in doc.tables]}


def save_structure_overlay(doc, path):
    """Section boxes (blue) and tables (green, with column and row lines) drawn on the page image."""
    from .dossier import redact_ids                    # same privacy rule as every evidence image (Aadhaar digits blacked out)
    im = Image.open(doc.image_path).convert("RGB")
    redact_ids(doc, im)
    d = ImageDraw.Draw(im)
    for s in doc.sections:
        d.rectangle(s["bbox"], outline=(40, 90, 200), width=3)
        d.text((s["bbox"][0] + 4, max(0, s["bbox"][1] - 14)), f"{s['sid']} {s['title']}"[:50], fill=(40, 90, 200))
    for t in doc.tables:
        d.rectangle(t["bbox"], outline=(20, 150, 60), width=4)
        for c in t["columns"]:
            d.line([(c["x0"], t["bbox"][1]), (c["x0"], t["bbox"][3])], fill=(20, 150, 60), width=1)
        for r in t["rows"]:
            d.line([(t["bbox"][0], r["bbox"][1] - 2), (t["bbox"][2], r["bbox"][1] - 2)], fill=(120, 200, 140), width=1)
        d.text((t["bbox"][0] + 4, max(0, t["bbox"][1] - 14)), f"{t['tid']} {t['kind']} ({len(t['rows'])} rows)", fill=(20, 150, 60))
    im.save(path)
    return path
