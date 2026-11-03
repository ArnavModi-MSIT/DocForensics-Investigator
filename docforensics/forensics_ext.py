"""Visual forensics, part 2 (5.4): typography / spacing / alignment, image insertions (JPEG block grid), photo and signature regions.

Everything here is an INDICATOR, computed relative to the page's own content, and every check reports why it did not run
instead of implying a clean result.

1. Typography, spacing, alignment (from OCR word boxes only; needs Tesseract-quality boxes)
   per field value: text height, character width / height (same character class: digits vs letters), baseline offset against its
   own label; column alignment of values whose labels share a column; irregular line spacing around a single line.
2. JPEG block grid: a JPEG-compressed page has an 8x8 blocking pattern with one phase. Content pasted in from elsewhere, resized
   or recompressed with a different offset has another phase. Tiles whose strong grid disagrees with the page's dominant grid
   are grouped into regions. Only meaningful when the page was not resampled after JPEG compression (no deskew, not a PDF render).
3. Photo region (ID cards): located as a continuous-tone, non-text, photo-shaped block (QR codes and barcodes are rejected as
   two-tone). Its grid phase is compared with the page. Photo identity is NOT matched.
4. Signature region: located near a 'Signature / Signatory / Signed' label; checked for presence, overlap with printed text and for
   paper-tone / grain / compression that differs from the page. Signature authenticity is NOT judged.
"""
import re
import cv2
import numpy as np
from rapidfuzz import fuzz
from .schema import Finding
from .ingest import read_gray, read_rgb
from .forensics import flatten_illumination, ela_map, _metrics, _robust_z, FLOORS, THRESH, FAMILY, FLAGGING, BGVAR_MAX

# ---------------------------------------------------------------------------------------------------- 1. typography
# (weak, strong) robust-z thresholds, calibrated on the synthetic benchmark (see ARCHITECTURE.md)
# OCR-box signals (height / width / baseline) are noisy (clean fields reach z 7-10), hence the high thresholds.
# The pixel signals (stroke weight = edge perimeter per ink pixel, glyph width) are REPORTED in the z-table but never used for flagging:
# they were stable on the single-style benchmark pages, but real documents set names, totals and headings in bold, so bold vs regular
# looks exactly like a retyped value (a genuine fictional payslip produced 7 false findings). They stay visible for a human reviewer.
TYPO_THRESH = {"height": (6.0, 10.0), "width": (6.0, 10.0), "baseline": (6.0, 10.0), "weight": (4.5, 6.0), "glyph_width": (4.5, 6.0)}
TYPO_FLOORS = {"height": (0.10, 2.0), "width": (0.07, 0.02), "baseline": (0.0, 0.12), "weight": (0.06, 1e-3), "glyph_width": (0.06, 1e-3)}
PIXEL_SIGNALS = ("weight", "glyph_width")


def _glyph_feats(gray, bb, pad=3):
    """(edge perimeter / ink pixels, median glyph width) of a value region; None when there is too little ink."""
    h, w = gray.shape
    x0, y0, x1, y1 = bb
    g = gray[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
    if g.size < 200:
        return None
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (g < t).astype(np.uint8)
    if ink.sum() < 30:
        return None
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    cc = [st[i] for i in range(1, n) if st[i][4] >= 8]
    if len(cc) < 3:
        return None
    cnts, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return sum(cv2.arcLength(c, True) for c in cnts) / ink.sum(), float(np.median([c[2] for c in cc]))


def _value_words(doc, f):
    x0, y0, x1, y1 = f.bbox
    return [w for w in doc.words if x0 - 2 <= (w.bbox[0] + w.bbox[2]) / 2 <= x1 + 2 and y0 - 2 <= (w.bbox[1] + w.bbox[3]) / 2 <= y1 + 2]


def _med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def _typo_rows(doc, gray=None):
    rows = []
    for f in doc.fields:
        if f.key in ("aadhaar_no",) or f.match == "pattern" or (":" in f.label.strip(" :") and not f.label.strip().endswith(":")):
            continue
        if any(t["bbox"][1] - 2 <= (f.bbox[1] + f.bbox[3]) / 2 <= t["bbox"][3] + 2 for t in doc.tables):
            continue                                                            # table cells follow table rules, not the form-column rule                                                            # label-free values, or a label glued to its value ('Income:Rs.')
        vws = _value_words(doc, f)
        if not vws:
            continue
        text = " ".join(w.text for w in vws)
        h = _med([w.bbox[3] - w.bbox[1] for w in vws])
        cy = _med([(w.bbox[1] + w.bbox[3]) / 2 for w in vws])
        digits = sum(c.isdigit() or c in "/-.,:" for c in text)
        lab = [w for w in doc.words if w not in vws and abs((w.bbox[1] + w.bbox[3]) / 2 - cy) <= 0.6 * h
               and w.bbox[2] <= f.bbox[0] + 3 and w.bbox[0] >= f.bbox[0] - 700]
        lab = sorted(lab, key=lambda w: -w.bbox[2])[:4]
        off = ((cy - _med([(w.bbox[1] + w.bbox[3]) / 2 for w in lab])) / h) if lab else None
        gf = _glyph_feats(gray, f.bbox) if gray is not None else None
        rows.append({"f": f, "h": h, "weight": gf[0] if gf else None, "gw": gf[1] if gf else None, "cw": (f.bbox[2] - f.bbox[0]) / max(1, len(text)) / max(1, h), "cls": "num" if digits >= 0.7 * len(text) else "txt",
                     "off": off, "lab_x0": min(w.bbox[0] for w in lab) if lab else None, "lab_x1": max(w.bbox[2] for w in lab) if lab else None, "val_x0": f.bbox[0], "val_x1": f.bbox[2], "cy": cy})
    return rows


def analyze_typography(doc, audit, id_start=1, gray=None):
    """Returns (findings, table). OCR-box signals are skipped (with a reason) when word boxes are estimates (RapidOCR); the pixel
    signals still run. Needs >= 4 comparable fields."""
    boxes_ok = not str(doc.facts.get("ocr_engine", "")).startswith("rapid")
    rows = _typo_rows(doc, gray)
    if len(rows) < 4:
        doc.facts["typography"] = "not run: too few labelled fields to compare (need >= 4)"
        return [], []
    findings, table, n = [], [], id_start
    for i, r in enumerate(rows):
        z = {}
        if boxes_ok:
            z["height"], _ = _robust_z([x["h"] for x in rows], i, *TYPO_FLOORS["height"])
            same = [j for j, x in enumerate(rows) if x["cls"] == r["cls"]]
            if len(same) >= 4:
                z["width"], _ = _robust_z([rows[j]["cw"] for j in same], same.index(i), *TYPO_FLOORS["width"])
            lab = [j for j, x in enumerate(rows) if x["off"] is not None]
            if r["off"] is not None and len(lab) >= 4:
                z["baseline"], _ = _robust_z([rows[j]["off"] for j in lab], lab.index(i), *TYPO_FLOORS["baseline"])
        for sig, key in (("weight", "weight"), ("glyph_width", "gw")):
            px = [j for j, x in enumerate(rows) if x[key] is not None]
            if r[key] is not None and len(px) >= 4:
                z[sig], _ = _robust_z([rows[j][key] for j in px], px.index(i), *TYPO_FLOORS[sig])
        table.append({"doc_id": doc.doc_id, "field": r["f"].key, **{f"{k}_z": round(v, 1) for k, v in z.items()}})
        flags = [(k, abs(v)) for k, v in z.items() if k not in PIXEL_SIGNALS and abs(v) >= TYPO_THRESH[k][0]]
        strong = [x for x in flags if x[1] >= TYPO_THRESH[x[0]][1]]
        if len(flags) >= 2 or strong:
            f = r["f"]
            names = ", ".join(f"{k} (z={z[k]:+.1f})" for k, _ in flags)
            findings.append(Finding(
                fid=f"V-{n:03d}", kind="visual", title=f"Text style of '{f.label.strip(':')}' differs from the rest of {doc.doc_id}",
                severity="medium" if (len(flags) >= 2 or strong) and any(k in PIXEL_SIGNALS for k, _ in flags) else "low",
                confidence=round(min(0.8, 0.35 + 0.12 * len(flags) + (0.1 if any(k in PIXEL_SIGNALS for k, _ in flags) else 0)), 2), status="potential_manipulation",
                explanation=(f"The value {f.raw!r} differs from the other fields of the same page in {names}. A different text height, character width, baseline, "
                             "stroke weight or glyph width is typical when a value is retyped in another font, weight or size. A change of font FAMILY at the "
                             "same size (serif vs sans) is not reliably detectable. This is an indicator; compare against the original."),
                refs=[{"doc_id": doc.doc_id, "field": f.key, "value": f.raw, "bbox": list(f.bbox)}], metrics=z,
                method="typography peer analysis (OCR boxes)"))
            n += 1
    # column alignment: values whose labels share one left edge should share one value edge
    lrows = [r for r in rows if r["lab_x0"] is not None]
    if len(lrows) >= 4:
        h = _med([r["h"] for r in lrows])
        labx = _med([r["lab_x0"] for r in lrows])
        col = [r for r in lrows if abs(r["lab_x0"] - labx) <= 0.8 * h]
        if len(col) >= 4:
            ys = [r["cy"] for r in col]

            def fit(edge):
                xs = [r[edge] for r in col]
                slopes = [(xs[j] - xs[i]) / (ys[j] - ys[i]) for i in range(len(col)) for j in range(i + 1, len(col)) if abs(ys[j] - ys[i]) > 2 * h]
                slope = _med(slopes) if slopes else 0.0                          # tilt / perspective: the value column is a LINE, not a vertical
                icpt = _med([x - slope * y for x, y in zip(xs, ys)])
                return slope, [x - (icpt + slope * y) for x, y in zip(xs, ys)]
            fits = {e: fit(e) for e in ("val_x0", "val_x1")}                      # left-aligned text or right-aligned numbers: take the tighter edge
            edge = min(fits, key=lambda e: _med([abs(v) for v in fits[e][1]]))
            slope, res = fits[edge]
            if sum(abs(v) <= 0.8 * h for v in res) >= 0.8 * len(col):                 # a real, straight value column exists
                for r, v in zip(col, res):
                    expected = r[edge] - v
                    if edge == "val_x0" and r["lab_x1"] is not None and r["lab_x1"] + 0.6 * h >= expected:   # a long label pushes its value right
                        continue
                    if abs(v) > max(3.0 * h, 40):
                        f = r["f"]
                        findings.append(Finding(
                            fid=f"V-{n:03d}", kind="visual", title=f"Value of '{f.label.strip(':')}' is misaligned in {doc.doc_id}", severity="low",
                            confidence=0.5, status="observed",
                            explanation=(f"Most values of this form line up in one column, but {f.raw!r} sits {v:+.0f}px away from that column. "
                                         "A retyped or pasted value is often not placed exactly like the original."),
                            refs=[{"doc_id": doc.doc_id, "field": f.key, "value": f.raw, "bbox": list(f.bbox)}],
                            metrics={"offset_px": round(v, 1), "tilt_slope": round(slope, 4), "aligned_on": "left" if edge == "val_x0" else "right"}, method="column alignment check"))
                        n += 1
            # spacing: one line moved between its neighbours leaves one big and one small gap
            cs = sorted(col, key=lambda r: r["cy"])
            gaps = [b["cy"] - a["cy"] for a, b in zip(cs, cs[1:])]
            if len(gaps) >= 3:
                med = _med(gaps)
                for k in range(1, len(cs) - 1):
                    d1, d2 = gaps[k - 1] - med, gaps[k] - med
                    if d1 * d2 < 0 and min(abs(d1), abs(d2)) > max(0.45 * h, 6) and abs(abs(d1) - abs(d2)) < 0.5 * max(abs(d1), abs(d2)):
                        f = cs[k]["f"]
                        findings.append(Finding(
                            fid=f"V-{n:03d}", kind="visual", title=f"Irregular line spacing around '{f.label.strip(':')}' in {doc.doc_id}", severity="low",
                            confidence=0.5, status="observed",
                            explanation=(f"Lines are normally {med:.0f}px apart, but the line of {f.raw!r} sits {abs(d1):.0f}px off its place between its "
                                         "neighbours. A line that was edited and re-inserted can end up slightly shifted."),
                            refs=[{"doc_id": doc.doc_id, "field": f.key, "value": f.raw, "bbox": list(f.bbox)}],
                            metrics={"typical_gap": med, "gap_above": gaps[k - 1], "gap_below": gaps[k]}, method="line spacing check"))
                        n += 1
    doc.facts["typography"] = f"run on {len(rows)} fields"
    audit.log("typography", doc_id=doc.doc_id, fields=len(rows), flagged=[x.refs[0]["field"] for x in findings])
    return findings, table


# ---------------------------------------------------------------------------------------------------- 2. JPEG block grid
def grid_phase(gray, axis, small=12):
    """(phase 0-7, strength) of the 8-pixel blocking pattern along one axis. Only SMALL intensity steps are used (flat areas):
    text edges produce huge steps at random positions and would hide the faint step JPEG blocks leave at multiples of 8."""
    d = np.abs(np.diff(gray.astype(np.float32), axis=axis))
    idx = np.arange(d.shape[axis]) % 8
    prof = np.zeros(8)
    for k in range(8):
        sl = d[:, idx == k] if axis == 1 else d[idx == k, :]
        sl = sl[sl < small]
        if sl.size < 50:
            return 0, 0.0
        prof[k] = sl.mean()
    base = np.median(prof)
    return int(prof.argmax()), float((prof.max() - base) / max(base, 1e-3))


def _grid_usable(doc):
    if doc.born_digital:
        return "PDF page rendering (no JPEG grid)"
    if abs(doc.skew_deg) > 0.05:
        return "page was rotated for deskew (JPEG grid destroyed)"
    return None


GRID_TILE, GRID_MIN_STRENGTH = 64, 0.2
GRID_DOMINANT_MIN = 0.85      # share of tiles that must agree before the page is considered to have ONE grid
GRID_ODD_STRENGTH = 0.35      # a tile only counts as 'different grid' when its own grid is confident


def analyze_grid(doc, gray_raw, audit, id_start=1):
    """Tiles whose strong 8x8 grid has a different phase from the page's dominant one, grouped into regions."""
    why = _grid_usable(doc)
    if why:
        return [], [], why
    h, w = gray_raw.shape
    tiles = []
    for y in range(0, h - GRID_TILE + 1, GRID_TILE):
        for x in range(0, w - GRID_TILE + 1, GRID_TILE):
            t = gray_raw[y:y + GRID_TILE, x:x + GRID_TILE]
            (px, sx), (py, sy) = grid_phase(t, 1), grid_phase(t, 0)
            if min(sx, sy) >= GRID_MIN_STRENGTH:
                tiles.append((x, y, (px + x) % 8, (py + y) % 8, min(sx, sy)))         # phase relative to the page origin
    if len(tiles) < 20:
        return [], [], "too little textured content with a readable JPEG grid (image not JPEG, resampled or very smooth)"
    phases = [(a, b) for _, _, a, b, _ in tiles]
    dom = max(set(phases), key=phases.count)
    share = phases.count(dom) / len(phases)
    if share < GRID_DOMINANT_MIN:
        return [], [], f"no dominant JPEG grid (only {share * 100:.0f}% of tiles agree): page probably resampled"
    regions_found = []                                                     # one region = connected tiles sharing ONE common non-dominant grid
    for ph in sorted({(a, b) for _, _, a, b, st in tiles if (a, b) != dom}):
        mask = np.zeros((h // GRID_TILE + 1, w // GRID_TILE + 1), np.uint8)
        for x, y, a, b, st in tiles:
            if (a, b) == ph and st >= GRID_ODD_STRENGTH:
                mask[y // GRID_TILE, x // GRID_TILE] = 1
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        regions_found += [(ph, stats[i]) for i in range(1, n) if stats[i][4] >= 4]       # isolated odd tiles are noise
    regions, findings, k = [], [], id_start
    for ph, st_ in regions_found:
        cx, cy, cw, ch, area = st_
        bbox = [int(cx * GRID_TILE), int(cy * GRID_TILE), int((cx + cw) * GRID_TILE), int((cy + ch) * GRID_TILE)]
        regions.append({"kind": "grid", "bbox": bbox, "tiles": int(area)})
        inside = [f for f in doc.fields if bbox[0] <= (f.bbox[0] + f.bbox[2]) / 2 <= bbox[2] and bbox[1] <= (f.bbox[1] + f.bbox[3]) / 2 <= bbox[3]]
        findings.append(Finding(
            fid=f"V-{k:03d}", kind="visual", title=f"Region with a different JPEG block grid in {doc.doc_id}",
            severity="medium" if inside else "low", confidence=0.5 if inside else 0.4, status="potential_manipulation" if inside else "observed",
            explanation=(f"{share * 100:.0f}% of the textured page tiles share one 8x8 JPEG grid, but a block of {int(area)} tiles at {bbox} has another "
                         f"(phase {ph} instead of {dom}). Content that was pasted in, resized or came from a differently cropped JPEG typically carries its own grid. "
                         + (f"Fields inside: {', '.join(f.key for f in inside)}. " if inside else "No extracted field lies inside; this may be an inserted image or photo. ")
                         + "This is an indicator only; verify against the original."),
            refs=[{"doc_id": doc.doc_id, "field": inside[0].key if inside else None, "value": inside[0].raw if inside else None, "bbox": bbox}],
            metrics={"dominant_phase": list(dom), "agreeing_share": round(share, 2), "tiles": int(area)}, method="JPEG block-grid consistency"))
        k += 1
    return findings, regions, f"run on {len(tiles)} tiles; dominant grid {share * 100:.0f}%"


# ---------------------------------------------------------------------------------------------------- 3. photo region
def find_photo(doc, gray):
    """Continuous-tone, photo-shaped block away from printed text. Returns bbox or None (QR codes / barcodes rejected as two-tone)."""
    h, w = gray.shape
    text = np.zeros((h, w), np.uint8)
    for wd in doc.words:
        if wd.conf >= 0.3:
            x0, y0, x1, y1 = wd.bbox
            cv2.rectangle(text, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), 1, -1)
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    m = cv2.blur(g, (21, 21)); sd = np.sqrt(np.maximum(cv2.blur(g * g, (21, 21)) - m * m, 0))       # wide window: portraits are smooth but not flat
    cand = ((sd > 6) & (text == 0)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if not (0.012 * h * w <= cw * ch <= 0.2 * h * w) or not (0.55 <= cw / ch <= 1.2) or area / (cw * ch) < 0.55:
            continue
        reg = gray[y:y + ch, x:x + cw]
        if ((reg < 50) | (reg > 205)).mean() > 0.85:                        # two-tone: QR code / barcode / logo, not a photograph
            continue
        if best is None or cw * ch > best[2] * best[3]:
            best = (int(x), int(y), int(cw), int(ch))
    return None if best is None else [best[0], best[1], best[0] + best[2], best[1] + best[3]]


def analyze_photo(doc, gray_raw, audit, id_start=1):
    if doc.doc_type != "id_card":
        return [], [], "not an ID card"
    gray = flatten_illumination(gray_raw)
    bb = find_photo(doc, gray)
    if bb is None:
        return [], [], "no photo-shaped continuous-tone region located (busy background, no photo, or photo too small)"
    reg = {"kind": "photo", "bbox": bb}
    why = _grid_usable(doc)
    x0, y0, x1, y1 = bb
    if why:
        reg["grid"] = f"not checked: {why}"
        return [], [reg], f"photo located at {bb}; grid check skipped ({why})"
    inner = gray_raw[y0:y1, x0:x1]
    (px, sx), (py, sy) = grid_phase(inner, 1), grid_phase(inner, 0)
    page = gray_raw.copy(); page[y0 - 4:y1 + 4, x0 - 4:x1 + 4] = 128
    mask = np.ones(page.shape, bool); mask[y0 - 4:y1 + 4, x0 - 4:x1 + 4] = False
    tiles = []
    for yy in range(0, gray_raw.shape[0] - GRID_TILE + 1, GRID_TILE):
        for xx in range(0, gray_raw.shape[1] - GRID_TILE + 1, GRID_TILE):
            if not mask[yy:yy + GRID_TILE, xx:xx + GRID_TILE].all():
                continue
            t = gray_raw[yy:yy + GRID_TILE, xx:xx + GRID_TILE]
            (a, s1), (b, s2) = grid_phase(t, 1), grid_phase(t, 0)
            if min(s1, s2) >= GRID_MIN_STRENGTH:
                tiles.append(((a + xx) % 8, (b + yy) % 8))
    reg["grid"] = "no strong grid inside the photo" if min(sx, sy) < GRID_MIN_STRENGTH else f"photo grid phase ({(px + x0) % 8},{(py + y0) % 8}), strength {min(sx, sy):.2f}"
    if len(tiles) < 15 or min(sx, sy) < GRID_MIN_STRENGTH:
        return [], [reg], f"photo located at {bb}; grid not conclusive"
    dom = max(set(tiles), key=tiles.count)
    if tiles.count(dom) / len(tiles) < 0.6:
        return [], [reg], f"photo located at {bb}; page grid not dominant"
    ph = ((px + x0) % 8, (py + y0) % 8)
    reg["page_grid"] = list(dom)
    if ph == dom:
        return [], [reg], f"photo located at {bb}; its JPEG grid matches the page"
    f = Finding(fid=f"V-{id_start:03d}", kind="visual", title=f"Photo region on {doc.doc_id} has a different JPEG block grid than the rest of the card",
                severity="medium", confidence=0.6, status="potential_manipulation",
                explanation=(f"The photo (box {bb}) shows an 8x8 compression grid with phase {ph}, while {tiles.count(dom) / len(tiles) * 100:.0f}% of the rest of the page "
                             f"shares phase {dom}. A photo that was replaced or inserted after the page was last saved often keeps its own grid. "
                             "Photo identity is not compared, and re-saved or resized cards can also differ; verify against the original."),
                refs=[{"doc_id": doc.doc_id, "field": None, "value": None, "bbox": bb}], metrics={"photo_phase": list(ph), "page_phase": list(dom)},
                method="photo-region JPEG grid check")
    return [f], [reg], f"photo located at {bb}; grid differs from the page"


# ---------------------------------------------------------------------------------------------------- 4. signatures
SIG_WORDS = ["signature", "signatory", "signed", "sign"]


def _border_step(gray, bbox, words, pad=8, ring=30):
    """(tone step, grain ratio) between the background just INSIDE a region and the ring of paper just OUTSIDE it (ink and printed
    words excluded). A region pasted from elsewhere often brings its own paper tone and grain, i.e. a step at its border."""
    H, W = gray.shape
    x0, y0, x1, y1 = bbox
    bx = lambda p: (max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p))
    ax0, ay0, ax1, ay1 = bx(pad)
    ox0, oy0, ox1, oy1 = bx(pad + ring)
    win = gray[oy0:oy1, ox0:ox1]
    thr = float(np.median(win)) - 30
    ink = (win < thr).astype(np.uint8)
    ok = cv2.dilate(ink, np.ones((7, 7), np.uint8)) == 0                    # keep clear of strokes and their anti-aliased edges
    for wd in words:
        if wd.conf >= 0.3:
            a, b, c, d = wd.bbox
            ok[max(0, b - 4 - oy0):max(0, d + 4 - oy0), max(0, a - 4 - ox0):max(0, c + 4 - ox0)] = False
    inner = np.zeros(win.shape, bool); inner[ay0 - oy0:ay1 - oy0, ax0 - ox0:ax1 - ox0] = True
    res = win.astype(np.float32) - cv2.medianBlur(win, 3).astype(np.float32)
    a_in, a_out = ok & inner, ok & ~inner
    if a_in.sum() < 300 or a_out.sum() < 300:
        return None
    return float(np.median(win[a_in]) - np.median(win[a_out])), float((res[a_in].std() + 1e-3) / (res[a_out].std() + 1e-3))


def _sig_labels(doc):
    out = []
    for w in doc.words:
        t = re.sub(r"[^a-z]", "", w.text.lower())
        if len(t) >= 4 and max(fuzz.ratio(t, s) for s in SIG_WORDS[:3]) >= 82 or t == "signed":
            line = [x for x in doc.words if abs((x.bbox[1] + x.bbox[3]) / 2 - (w.bbox[1] + w.bbox[3]) / 2) <= 0.6 * (w.bbox[3] - w.bbox[1])
                    and abs(x.bbox[0] - w.bbox[0]) <= 400]
            if t == "signed" and any(re.search(r"\d", x.text) for x in line if x.bbox[0] > w.bbox[0]):      # 'Signed: 31 Oct 2026' states WHEN, it is not a signature field
                continue
            out.append({"word": w, "bbox": [min(x.bbox[0] for x in line), min(x.bbox[1] for x in line), max(x.bbox[2] for x in line), max(x.bbox[3] for x in line)]})
    return out


def _typed_above(doc, lb, h):
    """A typed or printed NAME (up to four capitalised words, narrow) just above a signature label: electronic documents sign with a typed
    name, which is a mark but not handwriting. Returns its box or None. Sentences, lowercase text and anything further up do not count."""
    x0, y0, x1 = lb[0] - 2 * h, lb[1] - 3.5 * h, lb[2] + 12 * h
    ws = [w for w in doc.words if w.conf >= 0.5 and re.search(r"[A-Za-z]{2}", w.text) and x0 <= (w.bbox[0] + w.bbox[2]) / 2 <= x1
          and y0 <= (w.bbox[1] + w.bbox[3]) / 2 < lb[1] and w.bbox[3] <= lb[3]]
    if not ws or len(ws) > 4 or any(not (w.text[:1].isupper() or w.text[:1].isdigit()) for w in ws):
        return None
    box = [min(w.bbox[0] for w in ws), min(w.bbox[1] for w in ws), max(w.bbox[2] for w in ws), max(w.bbox[3] for w in ws)]
    if box[2] - box[0] > 14 * h or box[3] - box[1] > 3 * max(h, max(w.bbox[3] - w.bbox[1] for w in ws)):
        return None
    return box


def _ink_blob(gray, doc, win, label_bbox):
    """Largest handwriting-like ink component inside `win` after removing printed words and ruled lines."""
    x0, y0, x1, y1 = [int(v) for v in win]
    H, W = gray.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 30 or y1 - y0 < 10:
        return None
    reg = gray[y0:y1, x0:x1].copy()
    thr = max(60, min(200, float(np.median(reg)) - 45))
    ink = (reg < thr).astype(np.uint8)
    for wd in doc.words:                                                    # printed words (not handwriting) are removed
        a, b, c, d = wd.bbox
        if wd.conf >= 0.5 and not (a < label_bbox[2] and c > label_bbox[0] and b < label_bbox[3] and d > label_bbox[1] and wd.text.lower().startswith("sign")):
            ink[max(0, b - y0 - 2):max(0, d - y0 + 2), max(0, a - x0 - 2):max(0, c - x0 + 2)] = 0
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))   # signature rules / underlines
    ink = cv2.subtract(ink, cv2.dilate(lines, np.ones((3, 3), np.uint8)))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area >= 150 and cw >= 30 and ch >= 10 and 0.04 <= area / (cw * ch) <= 0.75 and (best is None or area > best[4]):
            best = (x, y, cw, ch, area)
    return None if best is None else [x0 + best[0], y0 + best[1], x0 + best[0] + best[2], y0 + best[1] + best[3], int(best[4])]


def _unlabelled_blobs(gray, doc):
    """Wide, handwriting-like marks in the lower part of the page when no signature label exists (certificates often have only a name and
    a title under the line). Seals/logos (near-square), borders and ruled lines are rejected; printed words are removed first."""
    H, W = gray.shape
    y0 = int(H * 0.5)
    reg = gray[y0:, :].copy()
    ink = (reg < max(60, min(200, float(np.median(reg)) - 45))).astype(np.uint8)
    for wd in doc.words:
        if wd.conf >= 0.5:
            a, b, c, d = wd.bbox
            ink[max(0, b - y0 - 4):max(0, d - y0 + 4), max(0, a - 4):max(0, c + 4)] = 0
    for k in ((60, 1), (1, 60)):
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
        ink = cv2.subtract(ink, cv2.dilate(lines, np.ones((3, 3), np.uint8)))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, cw, ch, area = st[i]
        if (area >= 300 and cw >= 60 and 15 <= ch <= 170 and cw / ch >= 1.5 and 0.04 <= area / (cw * ch) <= 0.5
                and x > 8 and x + cw < W - 8):
            out.append([int(x), int(y + y0), int(x + cw), int(y + y0 + ch), int(area)])
    return sorted(out, key=lambda b: -b[4])[:2]


def analyze_signature(doc, gray_raw, audit, id_start=1, allow_unlabelled=False):
    labels = _sig_labels(doc)
    if not labels and allow_unlabelled:
        return _analyze_unlabelled(doc, gray_raw, id_start)
    if not labels:
        return [], [], "no signature label on the page"
    gray = flatten_illumination(gray_raw)
    findings, regions, n = [], [], id_start

    labels_words = [l["word"] for l in labels]
    for lab in labels[:3]:
        lb = lab["bbox"]
        h = max(10, lb[3] - lb[1])
        wins = {"above the label": (lb[0] - 2 * h, lb[1] - 7 * h, lb[2] + 12 * h, lb[1] + 0.2 * h),
                "right of the label": (lb[2], lb[1] - 3 * h, lb[2] + 14 * h, lb[3] + 3 * h),
                "below the label": (lb[0] - 2 * h, lb[3] - 0.2 * h, lb[2] + 12 * h, lb[3] + 6 * h)}
        found = None
        for place, win in wins.items():
            blob = _ink_blob(gray, doc, win, lb)
            if blob and (found is None or blob[4] > found[1][4] * 1.5):
                found = (place, blob)
        if not found:
            typed = _typed_above(doc, lb, h)
            if typed:                                                     # typed name at the signature position: present, not handwriting
                regions.append({"kind": "signature", "bbox": typed, "found": True, "placement": "typed name above the label", "typed": True})
                continue
            regions.append({"kind": "signature", "bbox": lb[:4], "found": False})
            findings.append(Finding(fid=f"V-{n:03d}", kind="visual", title=f"No signature found near its label in {doc.doc_id}", severity="low", confidence=0.5,
                                    status="observed",
                                    explanation=f"A signature label ({lab['word'].text!r}) is printed at {lb[:4]}, but no handwriting-like ink was found above, beside or below it. "
                                                "The document may be unsigned, or the signature is too faint to detect. Verify against the original.",
                                    refs=[{"doc_id": doc.doc_id, "field": None, "value": None, "bbox": lb[:4]}], method="signature presence check"))
            n += 1
            continue
        place, blob = found
        sb = blob[:4]
        area = (sb[2] - sb[0]) * (sb[3] - sb[1])
        over, pad = 0, 8                                                  # printed ink was removed from the blob, so compare its padded box
        pb = [sb[0] - pad, sb[1] - pad, sb[2] + pad, sb[3] + pad]
        area = (pb[2] - pb[0]) * (pb[3] - pb[1])
        for wd in doc.words:
            if wd.conf >= 0.5 and not wd.text.lower().startswith("sign") and wd.bbox != lab["word"].bbox and wd not in labels_words:
                ox = max(0, min(pb[2], wd.bbox[2]) - max(pb[0], wd.bbox[0])); oy = max(0, min(pb[3], wd.bbox[3]) - max(pb[1], wd.bbox[1]))
                over += ox * oy
        reg = {"kind": "signature", "bbox": sb, "found": True, "placement": place, "ink_pixels": blob[4], "overlap_with_text": round(over / max(1, area), 2)}
        regions.append(reg)
        if over / max(1, area) >= 0.15:
            findings.append(Finding(fid=f"V-{n:03d}", kind="visual", title=f"Signature overlaps printed text in {doc.doc_id}", severity="medium", confidence=0.6,
                                    status="observed",
                                    explanation=f"The signature ({place}, box {sb}) covers {over / max(1, area) * 100:.0f}% printed text. A signature placed over typed content "
                                                "can indicate that it was added after the text, or the page was altered around it. Verify against the original.",
                                    refs=[{"doc_id": doc.doc_id, "field": None, "value": None, "bbox": sb}], metrics=reg, method="signature placement check"))
            n += 1
        # paper tone / grain step at the signature's border, judged against the same measure on the page's ordinary fields
        step = _border_step(gray_raw, sb, doc.words)
        peers = [x for x in (_border_step(gray_raw, b_, doc.words) for b_ in _peer_boxes(doc)) if x]
        if step and len(peers) >= 4:
            tone = np.array([p[0] for p in peers]); grain = np.log(np.array([p[1] for p in peers]))
            zt = (step[0] - np.median(tone)) / max(1.4826 * np.median(np.abs(tone - np.median(tone))), 1.0)
            zg = (np.log(step[1]) - np.median(grain)) / max(1.4826 * np.median(np.abs(grain - np.median(grain))), 0.25)
            reg["border_step"] = {"tone": round(step[0], 1), "grain_ratio": round(step[1], 2), "tone_z": round(float(zt), 1), "grain_z": round(float(zg), 1)}
            hit = [n_ for n_, ok in (("tone", abs(zt) >= 4 and abs(step[0]) >= 2.5), ("grain", abs(zg) >= 4 and not 0.6 <= step[1] <= 1.7)) if ok]
            if hit:
                findings.append(Finding(fid=f"V-{n:03d}", kind="visual", title=f"Signature region differs from the surrounding paper in {doc.doc_id}", severity="medium", confidence=0.55,
                                        status="potential_manipulation",
                                        explanation=(f"The background of the signature (box {sb}) differs from the paper just around it in {' and '.join(hit)} "
                                                     f"(tone step {step[0]:+.1f} levels, grain ratio {step[1]:.2f}), much more than the page's ordinary fields do. "
                                                     "A signature pasted in as an image often brings its own background. Signature authenticity is not assessed."),
                                        refs=[{"doc_id": doc.doc_id, "field": None, "value": None, "bbox": sb}], metrics=reg["border_step"],
                                        method="signature border consistency"))
                n += 1
    return findings, regions, f"{len(labels)} signature label(s)"


def _peer_boxes(doc):
    fb = [f.bbox for f in doc.fields]
    if len(fb) >= 4:
        return fb
    return [w.bbox for w in doc.words if w.conf >= 0.6 and w.bbox[2] - w.bbox[0] >= 40][:40]        # no fields: printed words are the controls


def _analyze_unlabelled(doc, gray_raw, n):
    gray = flatten_illumination(gray_raw)
    blobs = _unlabelled_blobs(gray, doc)
    if not blobs:
        return [], [], "no signature label and no handwriting-like mark found in the lower half"
    findings, regions = [], []
    peers = [x for x in (_border_step(gray_raw, b, doc.words) for b in _peer_boxes(doc)) if x]
    for b in blobs:
        sb = b[:4]
        reg = {"kind": "signature", "bbox": sb, "found": True, "labelled": False, "placement": "no label (handwriting-like mark, lower half)", "ink_pixels": b[4]}
        regions.append(reg)
        pb = [sb[0] - 8, sb[1] - 8, sb[2] + 8, sb[3] + 8]
        area = (pb[2] - pb[0]) * (pb[3] - pb[1])
        over = sum(max(0, min(pb[2], w.bbox[2]) - max(pb[0], w.bbox[0])) * max(0, min(pb[3], w.bbox[3]) - max(pb[1], w.bbox[1]))
                   for w in doc.words if w.conf >= 0.5)
        reg["overlap_with_text"] = round(over / max(1, area), 2)
        if over / max(1, area) >= 0.15:
            findings.append(Finding(fid=f"V-{n:03d}", kind="visual", title=f"Handwriting-like mark overlaps printed text in {doc.doc_id}", severity="low", confidence=0.4,
                                    status="observed",
                                    explanation=f"A mark that looks like a signature (box {sb}) covers {over / max(1, area) * 100:.0f}% printed text. There is no signature label, so this is only an observation.",
                                    refs=[{"doc_id": doc.doc_id, "field": None, "value": None, "bbox": sb}], metrics=reg, method="signature placement check (unlabelled)"))
            n += 1
        step = _border_step(gray_raw, sb, doc.words)
        if step and len(peers) >= 6:
            tone = np.array([p[0] for p in peers])
            zt = (step[0] - np.median(tone)) / max(1.4826 * np.median(np.abs(tone - np.median(tone))), 1.0)
            reg["border_step"] = {"tone": round(step[0], 1), "tone_z": round(float(zt), 1)}
            if abs(zt) >= 4 and abs(step[0]) >= 2.5:
                findings.append(Finding(fid=f"V-{n:03d}", kind="visual", title=f"Signature-like mark differs from the surrounding paper in {doc.doc_id}", severity="low", confidence=0.4,
                                        status="potential_manipulation",
                                        explanation=(f"The background of the mark (box {sb}) is {step[0]:+.1f} grey levels away from the paper just around it, much more than ordinary "
                                                     "printed text is. An image of a signature placed on the page often brings its own background. No signature label exists, so "
                                                     "treat this as weak evidence. Authenticity is not assessed."),
                                        refs=[{"doc_id": doc.doc_id, "field": None, "value": None, "bbox": sb}], metrics=reg["border_step"], method="signature border consistency (unlabelled)"))
                n += 1
    return findings, regions, f"no signature label; {len(blobs)} handwriting-like mark(s) located"


def analyze_signature_only(doc, audit, id_start=1):
    """For pages that are not a supported type: only signature presence / placement / background (labelled or not)."""
    try:
        f, regs, note = analyze_signature(doc, read_gray(doc), audit, id_start, allow_unlabelled=True)
    except Exception as e:
        doc.facts["signature"] = f"failed: {type(e).__name__}"
        return []
    doc.facts["signature"], doc.facts["regions"] = note, regs
    audit.log("visual_ext", doc_id=doc.doc_id, signature=note, flagged=[x.fid for x in f])
    return f


# ---------------------------------------------------------------------------------------------------- driver
def analyze_ext(doc, audit, id_start=1):
    """All checks of this module for one recognised page. Never raises. Returns findings; fills doc.facts['typography'],
    doc.facts['regions'] and doc.facts['grid']; typography z-table in doc.facts['typography_table']."""
    findings, regions = [], []
    n = id_start
    try:
        f, table = analyze_typography(doc, audit, n, gray=flatten_illumination(read_gray(doc)))
        findings += f; n += len(f)
        doc.facts["typography_table"] = table
    except Exception as e:
        doc.facts["typography"] = f"failed: {type(e).__name__}"
    try:
        gray_raw = read_gray(doc)
        for key, fn in (("grid", analyze_grid), ("photo", analyze_photo), ("signature", analyze_signature)):
            f, regs, note = fn(doc, gray_raw, audit, n)
            findings += f; regions += regs; n += len(f)
            doc.facts[key] = note
    except Exception as e:
        doc.facts["regions_error"] = f"{type(e).__name__}: {str(e)[:60]}"
    doc.facts["regions"] = regions
    audit.log("visual_ext", doc_id=doc.doc_id, typography=doc.facts.get("typography"), grid=doc.facts.get("grid"), photo=doc.facts.get("photo"),
              signature=doc.facts.get("signature"), flagged=[x.fid for x in findings])
    return findings


def save_regions_overlay(doc, path):
    """Photo (magenta), signature (cyan) and grid-inconsistency (orange) regions drawn on the page; Aadhaar digits redacted."""
    from PIL import Image, ImageDraw
    from .dossier import redact_ids
    im = Image.open(doc.image_path).convert("RGB")
    redact_ids(doc, im)
    d = ImageDraw.Draw(im)
    col = {"photo": (200, 0, 200), "signature": (0, 160, 200), "grid": (240, 140, 0), "patch": (220, 0, 0)}
    for r in doc.facts.get("regions", []):
        c = col.get(r["kind"], (0, 0, 0))
        d.rectangle(r["bbox"], outline=c, width=4)
        d.text((r["bbox"][0] + 4, max(0, r["bbox"][1] - 14)), r["kind"] + ("" if r.get("found", True) else " (not found)"), fill=c)
    im.save(path)
    return path
