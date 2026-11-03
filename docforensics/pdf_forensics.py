"""Forensics for born-digital PDF pages (visual forensic analysis, part 3).

A generated PDF has no scan grain, noise or JPEG blocks to compare, but it records exactly how every number was drawn. This module
reads that record (pdf_layer.py) and looks for what an edit leaves behind:

  font outlier      one amount is set in a different font family from all the other amounts on the page (a number typed over another
                    in a different typeface, e.g. a monospaced figure among proportional ones)
  replacement patch a bordered, differently coloured rectangle drawn on top of the panel behind a value (the usual way to cover an
                    original number before writing a new one)
  covered text      text that is still in the file but is hidden under a rectangle drawn later (the original value under a patch)
  hidden text       invisible, zero-opacity, microscopic or off-page text
  file history      several saved revisions, a modification date later than the creation date, or an image / online editor as producer

Every finding is an OBSERVATION about the file's structure. None of them proves that a value was changed, by whom, or why: legitimate
documents also contain badges, annotations and re-saved revisions. They point a reviewer at a place to look and are worded that way.
"""
import re
from collections import Counter
from datetime import timedelta
from .schema import Finding
from .pdf_layer import parse_pdf_date, SCALE

CCY = r"(?:INR|Rs\.?|₹)"
MONEY_RE = re.compile(rf"^{CCY}\s*\d[\d,]*(?:\.\d{{1,2}})?$|^\d{{1,3}}(?:,\d{{2,3}})+(?:\.\d{{1,2}})?$", re.I)
EDITOR_RE = re.compile(r"(?i)photoshop|gimp|illustrator|canva|sejda|ilovepdf|smallpdf|pdfescape|pdf24|pdfcandy|pdf\s*editor|paint\.net|inkscape")
STYLE_WORDS = r"(bold|italic|oblique|regular|roman|light|medium|semibold|black|book|narrow|condensed)"


def family(font):
    """Font family without style words: 'AAAAAA+DejaVuSans-Bold' -> 'dejavusans'; 'DejaVuSansMono' stays distinct."""
    f = str(font).split("+")[-1]
    f = re.sub(rf"(?i)[-_ ,]?{STYLE_WORDS}", "", f)
    return f.strip("-_ ,").lower()


def _area(r):
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _inside(inner, outer, tol=1.0):
    return inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol


def _centre_in(b, r):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return r[0] <= cx <= r[2] and r[1] <= cy <= r[3]


def _cover(span, rect):
    """Share of the span's box covered by the rectangle."""
    w = min(span[2], rect[2]) - max(span[0], rect[0])
    h = min(span[3], rect[3]) - max(span[1], rect[1])
    return 0.0 if w <= 0 or h <= 0 else w * h / max(1e-9, _area(span))


def _col(c):
    return "none" if c is None else "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c)


def _px(r):
    return [round(v * SCALE, 1) for v in r]


def _field_in(doc, box_px):
    """The extracted field whose box centre lies in the region (so a visual finding can name the field it touches)."""
    for f in doc.fields:
        if _centre_in(f.bbox, box_px):
            return f
    return None


def money_spans(lay):
    seen, out = set(), []
    for s in lay["spans"]:
        t = s["text"].strip()
        key = (t, tuple(round(v) for v in s["bbox_pt"]))
        if key in seen:
            continue
        seen.add(key)
        if MONEY_RE.match(t) and sum(c.isdigit() for c in t) >= 3 and s["render"] != 3:
            out.append(s)
    return out


def font_outliers(lay, min_peers=4, majority=0.7, max_share=0.2):
    """Amounts whose font family differs from the family used by most other amounts on the page."""
    ms = money_spans(lay)
    if len(ms) < min_peers:
        return [], None, len(ms)
    fam = Counter(family(s["font"]) for s in ms)
    top, n_top = fam.most_common(1)[0]
    if n_top / len(ms) < majority:
        return [], top, len(ms)
    odd = [s for s in ms if family(s["font"]) != top]
    if not odd or len(odd) > max(1, int(max_share * len(ms))):
        return [], top, len(ms)
    return odd, top, len(ms)


def patches(lay, page_pt):
    """Bordered, filled rectangles lying on a larger filled panel of a clearly different colour and holding text. A rectangle whose
    style is repeated elsewhere on the page is a design element (badge, tile), not a one-off patch."""
    dr, spans, out = lay["drawings"], lay["spans"], []
    for f in dr:
        if f["fill"] is None or f["stroke"] is None:
            continue
        r = f["rect_pt"]
        w, h = r[2] - r[0], r[3] - r[1]
        if w < 8 or h < 6 or w > 0.7 * page_pt[0] or h > 0.2 * page_pt[1]:
            continue
        inside = [s for s in spans if _centre_in(s["bbox_pt"], r) and s["render"] != 3]
        if not inside:
            continue
        under = [g for g in dr if g is not f and g["fill"] is not None and g["seqno"] < f["seqno"] and _inside(r, g["rect_pt"]) and _area(g["rect_pt"]) > 1.3 * _area(r)]
        if not under:
            continue
        base = max(under, key=lambda g: g["seqno"])
        dist = sum(abs(a - b) for a, b in zip(f["fill"], base["fill"]))
        if dist < 0.06:
            continue
        if any(g is not f and g["fill"] == f["fill"] and g["stroke"] == f["stroke"] for g in dr):
            continue
        out.append({"draw": f, "panel": base, "spans": inside, "contrast": round(dist, 3)})
    return out


def covered_text(lay):
    """Text drawn BEFORE an opaque filled rectangle that covers most of it: still in the file, no longer visible."""
    out = []
    for s in lay["spans"]:
        if s["render"] == 3 or s["opacity"] == 0:
            continue
        for g in lay["drawings"]:
            if g["fill"] is None or g["seqno"] <= s["seqno"] or g["opacity"] < 0.95:
                continue
            if _cover(s["bbox_pt"], g["rect_pt"]) >= 0.8 and (s["color"] is None or sum(abs(a - b) for a, b in zip(s["color"], g["fill"])) > 0.15):
                out.append((s, g))
                break
    return out


def hidden_text(lay, page_pt):
    out = []
    for s in lay["spans"]:
        b = s["bbox_pt"]
        why = None
        if s["render"] == 3:
            why = "invisible text rendering mode"
        elif s["opacity"] == 0:
            why = "zero opacity"
        elif s["size"] and s["size"] < 1.5:
            why = f"microscopic font size ({s['size']} pt)"
        elif b[2] < 0 or b[3] < 0 or b[0] > page_pt[0] or b[1] > page_pt[1]:
            why = "placed outside the visible page"
        if why and len(s["text"].strip()) >= 2:
            out.append((s, why))
    return out


def analyze_pdf_page(doc, audit, n):
    """Findings for one born-digital page. Returns (findings, regions)."""
    lay = doc.pdf
    if not lay or not lay.get("spans"):
        return [], []
    page_pt = lay["page_pt"]
    findings, regions = [], []

    def fid():
        return f"V-{n + len(findings):03d}"

    odd, majority_font, n_money = font_outliers(lay)
    pats = patches(lay, page_pt)
    used_odd = set()
    for p in pats:                                                             # patch (+ the font of what is written on it)
        r = p["draw"]["rect_pt"]
        box = _px(r)
        odd_in = [s for s in odd if _centre_in(s["bbox_pt"], r)]
        used_odd.update(id(s) for s in odd_in)
        f = _field_in(doc, box)
        val = (odd_in[0]["text"] if odd_in else (p["spans"][0]["text"] if p["spans"] else "")).strip()
        strong = bool(odd_in)
        regions.append({"kind": "patch", "bbox": box, "found": True})
        expl = (f"The PDF draws a bordered rectangle (fill {_col(p['draw']['fill'])}, border {_col(p['draw']['stroke'])}) on top of the panel behind "
                f"{val!r}; the panel is {_col(p['panel']['fill'])}, so the rectangle stands out in colour as well as by its border. ")
        if strong:
            expl += (f"The figure on it is set in {odd_in[0]['font'].split('+')[-1]}, a different font family from the {n_money - 1} other amounts on the page "
                     f"({majority_font}). ")
        expl += ("A bordered patch with different lettering is a common way to cover a printed number and write another one. This is an observation about how the "
                 "file was built: it does not show that the value was changed, by whom or why, and a design element can look similar. Compare with the issuer's original.")
        findings.append(Finding(
            fid(), "visual", f"Possible replacement patch over {val!r} in {doc.doc_id}" if strong else f"Bordered rectangle of a different colour behind {val!r} in {doc.doc_id}",
            "medium" if strong else "low", 0.7 if strong else 0.4, "potential_manipulation" if strong else "observed", expl,
            [{"doc_id": doc.doc_id, "field": f.key if f else None, "value": val, "bbox": box, "exact": True}],
            metrics={"font": odd_in[0]["font"].split("+")[-1] if strong else None, "page_font": majority_font, "amounts_compared": n_money,
                     "panel_contrast": p["contrast"], "pdf_rect_pt": r},
            method="PDF structure: drawing order and fonts"))
    for s in odd:                                                              # font outlier without a patch
        if id(s) in used_odd:
            continue
        box = _px(s["bbox_pt"])
        f = _field_in(doc, box)
        regions.append({"kind": "patch", "bbox": box, "found": True})
        findings.append(Finding(
            fid(), "visual", f"Amount in a different font from the other amounts in {doc.doc_id}", "medium", 0.55, "potential_manipulation",
            f"{s['text']!r} is set in {s['font'].split('+')[-1]}, while {n_money - 1} other amounts on the page use {majority_font}. A figure typed over another "
            "in a different typeface can look like this. It can also be a deliberate emphasis; this is an indicator only, so compare with the original.",
            [{"doc_id": doc.doc_id, "field": f.key if f else None, "value": s["text"], "bbox": box, "exact": True}],
            metrics={"font": s["font"].split("+")[-1], "page_font": majority_font, "amounts_compared": n_money}, method="PDF structure: fonts"))
    for s, g in covered_text(lay):
        box = _px(s["bbox_pt"])
        f = _field_in(doc, box)
        regions.append({"kind": "patch", "bbox": box, "found": True})
        findings.append(Finding(
            fid(), "visual", f"Text hidden under a later-drawn rectangle in {doc.doc_id}", "high", 0.75, "potential_manipulation",
            f"The text {s['text']!r} is in the file but a filled rectangle drawn after it covers most of it, so it is not visible on the page. Text kept "
            "under a covering rectangle is how a value is hidden without deleting it. Read the hidden text and compare it with the visible one.",
            [{"doc_id": doc.doc_id, "field": f.key if f else None, "value": s["text"], "bbox": box, "exact": True}],
            metrics={"hidden_text": s["text"], "covering_fill": _col(g["fill"])}, method="PDF structure: drawing order"))
    for s, why in hidden_text(lay, page_pt)[:4]:
        box = _px(s["bbox_pt"])
        findings.append(Finding(
            fid(), "visual", f"Hidden text in {doc.doc_id}", "medium", 0.6, "observed",
            f"The text {s['text'][:60]!r} is present in the PDF but not visible ({why}). Invisible text can carry content that differs from what a reader sees "
            "(or be harmless, for example accessibility text). Read it and compare.",
            [{"doc_id": doc.doc_id, "field": None, "value": s["text"][:60], "bbox": None if why.startswith("placed") else box, "exact": True}],
            metrics={"reason": why}, method="PDF structure: text rendering"))
    sig = [s for s in lay["spans"] if re.search(r"(?i)signature|signatory", s["text"])]
    typed = 0
    for lab in sig:                                                            # a typed name close above (or beside) the label is a signature mark
        lb = lab["bbox_pt"]
        if any(s is not lab and lb[0] - 20 <= s["bbox_pt"][0] <= lb[2] and -6 <= lb[1] - s["bbox_pt"][3] <= 45 and not re.search(r"(?i)signature|signatory", s["text"])
               for s in lay["spans"]):
            typed += 1
    doc.facts["signature"] = (f"{typed} typed name(s) above {len(sig)} signature label(s); no handwriting in a generated PDF" if sig else "no signature label")
    doc.facts["pdf_checks"] = {"amount_fonts": f"{n_money} amounts compared (page font {majority_font})" if n_money else "too few amounts to compare",
                               "patches": len(pats), "covered_text": len(covered_text(lay)), "hidden_text": len(hidden_text(lay, page_pt))}
    return findings, regions


def analyze_pdf_file(docs, meta, n):
    """File-level history of one PDF (once per file). Returns findings (possibly none)."""
    notes = []
    if meta.get("revisions", 1) > 1:
        notes.append(f"saved {meta['revisions']} times (incremental updates): the file was changed after it was first created")
    c, m = parse_pdf_date(meta.get("created")), parse_pdf_date(meta.get("modified"))
    if c and m and m - c > timedelta(minutes=2):
        notes.append(f"modification date ({m:%Y-%m-%d %H:%M}) is later than the creation date ({c:%Y-%m-%d %H:%M})")
    prod = " / ".join(x for x in (meta.get("producer"), meta.get("creator")) if x)
    if prod and EDITOR_RE.search(prod):
        notes.append(f"produced by a graphics / online editing tool ({prod[:60]})")
    if meta.get("encrypted"):
        notes.append("the file is encrypted")
    if not notes:
        return []
    d = docs[0]
    return [Finding(f"V-{n:03d}", "visual", f"PDF file history shows later changes ({d.path.split('/')[-1].split(chr(92))[-1]})", "low", 0.45, "observed",
                    "The PDF's own records show that " + "; ".join(notes) + ". Re-saving is normal (signing, filling a form, optimising), so this alone proves nothing; "
                    "it matters only together with other findings on the same file.",
                    [{"doc_id": d.doc_id, "field": None, "value": None, "bbox": None}], metrics={"meta": {k: meta.get(k) for k in ("producer", "creator", "created", "modified", "revisions")}},
                    method="PDF metadata")]


def analyze_pdf_docs(docs, audit, id_start=1):
    """All born-digital pages: page findings, then file-level history once per file. Fills doc.facts['regions'] and ['pdf_checks']."""
    findings, seen_files, n = [], set(), id_start
    for d in docs:
        if not d.born_digital or not d.pdf:
            continue
        try:
            f, regs = analyze_pdf_page(d, audit, n)
        except Exception as e:                                                  # additive: never break the investigation
            d.facts["pdf_checks"] = f"failed: {type(e).__name__}"
            continue
        d.facts["regions"] = (d.facts.get("regions") or []) + regs
        findings += f
        n += len(f)
        if d.path not in seen_files:
            seen_files.add(d.path)
            meta = d.pdf.get("meta") or {}
            d.facts["pdf_meta"] = {k: meta.get(k) for k in ("producer", "creator", "created", "modified", "revisions", "pages")}
            g = analyze_pdf_file([x for x in docs if x.path == d.path], meta, n)
            findings += g
            n += len(g)
        audit.log("pdf_forensics", doc_id=d.doc_id, checks=d.facts.get("pdf_checks"), flagged=[x.fid for x in f])
    return findings
