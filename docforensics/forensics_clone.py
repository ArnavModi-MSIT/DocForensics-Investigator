"""Image-level checks that need no model: duplicated (cloned) regions and image metadata.

Duplicated region (copy-move). A pasted copy of a region (a digit, a stamp, a signature, a block of text) carries the SAME pixels, noise included,
as its source. Candidate pairs are found by matching low-frequency DCT features of overlapping blocks (sorted, so no all-pairs comparison); the
many pairs that share one shift vector are grouped; the group must cover a compact area and the two areas are then compared pixel by pixel
(normalised cross-correlation). Ordinary repeated text on a scan or photo does NOT pass the last test because its noise differs; a generated page
would (identical glyphs), so born-digital pages are not examined. The result is an observation: it says where two areas are identical, not why.

Image metadata. EXIF / PNG text of an image file: editing software named in the tags, or a modification date later than the capture date.
Most scans and screenshots carry no metadata at all, and that is not reported.
"""
import os, re
import numpy as np
from .schema import Finding

MAX_SIDE = 1800          # pages with a longer side are analysed at half size
BLOCK, STRIDE = 16, 4    # source blocks every STRIDE px; target blocks at every pixel, so a copy at ANY shift lines up with a source block
MIN_TEXTURE = 10.0       # standard deviation of a block below which it is blank paper
MIN_PAIRS = 40           # block pairs sharing one shift vector
MIN_SPAN = 40            # px (analysis scale): the matched area must be at least this wide and high
MIN_SHIFT = 32           # px: a region identical to its neighbour at a tiny shift is a smear, not a clone
STEP = 14.0              # quantisation step of the DCT features (two lattices, offset by half a step, so a value near a boundary still matches)
RES_NCC_MIN = 0.55       # correlation of the NOISE (page minus its blur): a copy brings its noise along, independent repeats do not
EDITOR_RE = re.compile(r"(?i)photoshop|gimp|lightroom|snapseed|picsart|pixlr|canva|affinity|paint\.net|illustrator|inkscape|fotor|facetune|meitu|photoscape")


def _dct_matrix(n):
    k = np.arange(n)[:, None]
    m = np.cos(np.pi * (2 * np.arange(n)[None, :] + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m.astype(np.float32)


def _features(gb, pts, D, chunk=40000):
    """Low-frequency AC DCT coefficients (8 per block) of the 16x16 blocks whose top-left corners are `pts` (x, y)."""
    W = np.lib.stride_tricks.sliding_window_view(gb, (BLOCK, BLOCK))
    D3 = D[:3]
    out = []
    for i in range(0, len(pts), chunk):
        p = pts[i:i + chunk]
        B = W[p[:, 1], p[:, 0]]                                                                  # (n, 16, 16) gathered in one step
        out.append(np.einsum("ij,njk,lk->nil", D3, B, D3).reshape(len(p), 9)[:, 1:])
    return np.concatenate(out) if out else np.zeros((0, 8), np.float32)


def _noise_ncc(g, gb, a, b, min_pixels=400):
    """Correlation of the NOISE of two same-sized boxes (x0, y0, x1, y1). The noise is the page minus its blur, taken only where both areas are flat
    (no strokes or edges): text edges are identical for identical glyphs and would hide the difference between a copy and a repeat. Returns -1
    when too few flat pixels exist to tell."""
    import cv2
    def flat(box):
        reg = gb[box[1]:box[3], box[0]:box[2]]
        m = cv2.blur(reg, (7, 7))
        return np.sqrt(np.maximum(cv2.blur(reg * reg, (7, 7)) - m * m, 0)) < 1.5
    fm = flat(a) & flat(b)
    if int(fm.sum()) < min_pixels:
        return -1.0
    ra = (g[a[1]:a[3], a[0]:a[2]] - gb[a[1]:a[3], a[0]:a[2]])[fm]
    rb = (g[b[1]:b[3], b[0]:b[2]] - gb[b[1]:b[3], b[0]:b[2]])[fm]
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / den) if den > 1e-6 else 0.0


def find_clones(gray):
    """Duplicated regions in a grayscale page: list of dicts {src, dst, pairs, ncc, shift} (boxes in pixels of `gray`)."""
    import cv2
    h0, w0 = gray.shape
    s = 0.5 if max(h0, w0) > MAX_SIDE else 1.0
    g = cv2.resize(gray, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else gray
    g = g.astype(np.float32)
    h, w = g.shape
    if h < 3 * BLOCK or w < 3 * BLOCK:
        return []
    gb = cv2.GaussianBlur(g, (0, 0), 1.2)
    m1 = cv2.boxFilter(gb, -1, (BLOCK, BLOCK), normalize=True, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    m2 = cv2.boxFilter(gb * gb, -1, (BLOCK, BLOCK), normalize=True, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    std = np.sqrt(np.maximum(m2 - m1 * m1, 0))[: h - BLOCK + 1, : w - BLOCK + 1]                  # block std at every top-left position
    ys, xs = np.nonzero(std >= MIN_TEXTURE)
    if len(xs) < 2 * MIN_PAIRS:
        return []
    if len(xs) > 120000:                                                                         # a very busy page: only the most textured positions
        top = np.argsort(-std[ys, xs])[:120000]
        ys, xs = ys[top], xs[top]
    pts = np.stack([xs, ys], axis=1)
    D = _dct_matrix(BLOCK)
    coef = _features(gb, pts, D)
    src = np.nonzero((pts[:, 0] % STRIDE == 0) & (pts[:, 1] % STRIDE == 0))[0]
    wts = np.array([1000003, 998244353, 1000033, 999983, 1000081, 1000099, 1000117, 1000121], np.int64)
    pairs_i, pairs_j = [], []
    for off in (0.0, STEP / 2):
        key = np.round((coef + off) / STEP).astype(np.int64)
        hsh = key @ wts
        order = np.argsort(hsh, kind="stable")
        hs = hsh[order]
        lo, hi = np.searchsorted(hs, hsh[src], "left"), np.searchsorted(hs, hsh[src], "right")
        for k in range(8):                                                                       # at most 8 partners per source block (flat or very common blocks are skipped)
            ok = (hi - lo > k) & (hi - lo <= 12)
            if not ok.any():
                break
            pairs_i.append(src[ok]); pairs_j.append(order[lo[ok] + k])
    if not pairs_i:
        return []
    I, J = np.concatenate(pairs_i), np.concatenate(pairs_j)
    d = pts[J] - pts[I]
    keep = (d[:, 0] ** 2 + d[:, 1] ** 2 >= MIN_SHIFT ** 2) & ((d[:, 0] > 0) | ((d[:, 0] == 0) & (d[:, 1] > 0)))      # one orientation per pair
    I, J, d = I[keep], J[keep], d[keep]
    if len(I) < MIN_PAIRS:
        return []
    code = (d[:, 0] + 20000) * 40000 + (d[:, 1] + 20000)
    uniq, inv, cnt = np.unique(code, return_inverse=True, return_counts=True)
    out = []
    for u in np.argsort(-cnt)[:6]:
        if cnt[u] < MIN_PAIRS:
            break
        sel = np.nonzero(inv == u)[0]
        sp = pts[I[sel]]
        dx, dy = int(d[sel[0], 0]), int(d[sel[0], 1])
        x0, y0 = sp.min(axis=0)
        x1, y1 = sp.max(axis=0) + BLOCK
        if x1 - x0 < MIN_SPAN or y1 - y0 < MIN_SPAN:
            continue
        if len(np.unique(sp, axis=0)) * STRIDE * STRIDE < 0.12 * (x1 - x0) * (y1 - y0):           # scattered matches (repeated glyphs), not one region
            continue
        tx0, ty0, tx1, ty1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
        if tx0 < 0 or ty0 < 0 or tx1 > w or ty1 > h or (min(x1, tx1) > max(x0, tx0) and min(y1, ty1) > max(y0, ty0)):
            continue                                                                              # target outside the page, or overlapping its source
        ncc = _noise_ncc(g, gb, (x0, y0, x1, y1), (tx0, ty0, tx1, ty1))
        if ncc >= RES_NCC_MIN:
            k = 1.0 / s
            out.append({"src": [int(x0 * k), int(y0 * k), int(x1 * k), int(y1 * k)], "dst": [int(tx0 * k), int(ty0 * k), int(tx1 * k), int(ty1 * k)],
                        "pairs": int(cnt[u]), "ncc": round(ncc, 3), "shift": [int(dx * k), int(dy * k)]})
    return out


def analyze_clone(doc, gray, audit, id_start=1):
    """Findings for one scanned or photographed page. Born-digital pages are skipped (identical glyphs are normal there)."""
    if doc.born_digital:
        return [], [], "not applicable (generated PDF)"
    clones = find_clones(gray)
    findings, regions = [], []
    for n, c in enumerate(clones[:3]):
        inside = [f for f in doc.fields for bx in (c["src"], c["dst"]) if bx[0] <= (f.bbox[0] + f.bbox[2]) / 2 <= bx[2] and bx[1] <= (f.bbox[1] + f.bbox[3]) / 2 <= bx[3]]
        regions += [{"kind": "clone", "bbox": c["src"], "found": True}, {"kind": "clone", "bbox": c["dst"], "found": True}]
        hit = inside[0] if inside else None
        findings.append(Finding(
            f"V-{id_start + n:03d}", "visual", f"Two identical image regions in {doc.doc_id}", "medium" if hit else "low", 0.5 if hit else 0.35,
            "potential_manipulation" if hit else "observed",
            f"The areas {c['src']} and {c['dst']} (shifted by {c['shift']}) carry the same pixels and the same noise (noise correlation {c['ncc']:.2f}, {c['pairs']} matching blocks). "
            "Independent print or scan noise makes even repeated characters differ, so an exact match suggests that one area was copied onto the other. "
            + (f"The area holds the field '{hit.key}'. " if hit else "No extracted field lies in them; it may be a stamp, logo or decoration duplicated by the issuer. ")
            + "This is an observation about the image, not a conclusion about intent; compare with the original.",
            [{"doc_id": doc.doc_id, "field": hit.key if hit else None, "value": hit.raw if hit else None, "bbox": c["dst"], "exact": True},
             {"doc_id": doc.doc_id, "field": None, "value": None, "bbox": c["src"], "exact": True}],
            metrics=c, method="duplicated-region (copy-move) check"))
    return findings, regions, (f"{len(clones)} duplicated region pair(s) found" if clones else "no duplicated regions")


# ---------------------------------------------------------------------------------------------------------------- image metadata
def read_image_meta(path):
    """{'software', 'datetime', 'datetime_original', 'make', 'model'} from EXIF or PNG text; empty when the file carries none."""
    out = {}
    try:
        from PIL import Image
        im = Image.open(path)
        ex = im.getexif()
        sub = {}
        try:
            sub = ex.get_ifd(0x8769)
        except Exception:
            pass
        for k, name in ((0x0131, "software"), (0x0132, "datetime"), (0x010F, "make"), (0x0110, "model")):
            if ex.get(k):
                out[name] = str(ex.get(k)).strip()
        if sub.get(0x9003):
            out["datetime_original"] = str(sub[0x9003]).strip()
        for k, v in (getattr(im, "text", None) or {}).items():                                         # PNG text chunks
            if k.lower() in ("software", "creation time", "comment"):
                out.setdefault("software" if k.lower() == "software" else k.lower(), str(v)[:120])
    except Exception:
        pass
    return out


def _exif_dt(s):
    from datetime import datetime
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt)
        except (ValueError, TypeError):
            pass
    return None


def image_meta_findings(doc, id_start):
    """One observation per image file when its metadata names an editing tool or shows a modification after capture."""
    meta = read_image_meta(doc.path)
    doc.facts["image_meta"] = meta
    notes = []
    if meta.get("software") and EDITOR_RE.search(meta["software"]):
        notes.append(f"the Software tag names an image editor ({meta['software'][:60]})")
    a, b = _exif_dt(meta.get("datetime_original", "")), _exif_dt(meta.get("datetime", ""))
    if a and b and (b - a).total_seconds() > 120:
        notes.append(f"the modification time ({b:%Y-%m-%d %H:%M}) is later than the capture time ({a:%Y-%m-%d %H:%M})")
    if not notes:
        return []
    return [Finding(f"V-{id_start:03d}", "visual", f"Image metadata shows editing in {os.path.basename(doc.path)}", "low", 0.4, "observed",
                    "The file's own metadata shows that " + "; ".join(notes) + ". Phones and scanners also rewrite metadata when a photo is rotated or cropped, so this alone proves nothing; "
                    "it matters together with other findings on the same file.",
                    [{"doc_id": doc.doc_id, "field": None, "value": None, "bbox": None}], metrics={"meta": meta}, method="image metadata")]
