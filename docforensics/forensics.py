"""Visual forensic analysis (5.4): per-field statistics vs. the document's own peers.

Signals (all computed on background pixels around the text so content differences
do not dominate):
  ela    - JPEG error-level (re-save q=90): single- vs double-compressed regions differ
  noise  - sensor/paper-grain residual (gray - median3): pasted patches are often too clean/noisy
  bglum  - mean background luminance: pasted patches rarely match paper tone exactly
  stroke - ink stroke width (distance transform): different font/rendering pipeline
A field is flagged only when it is an outlier relative to the other fields of the SAME
document (robust z-score, median/MAD). Page lighting is flattened first (phone photos).
"""
import cv2
import numpy as np
from .schema import Finding
from .ingest import read_gray, read_rgb

# (relative floor, absolute floor) for the robust scale of each metric
FLOORS = {"ela": (0.10, 0.10), "noise": (0.10, 0.20), "bglum": (0.0, 0.8), "inkdark": (0.0, 5.0), "stroke": (0.12, 0.25)}
FLAGGING = ("ela", "noise", "bglum", "inkdark")      # stroke is reported but not used for flagging (quantised at this DPI)
# (weak, strong) robust-z thresholds per signal, set from the distribution of innocent vs pasted fields on the
# synthetic benchmark (innocent max |z|: ela/noise ~6, paper tone ~1, ink darkness ~2; pasted: ink 7-12, tone 3-4.5).
# A field is flagged when >= 2 independent signal FAMILIES reach their weak threshold, or any one signal reaches its strong one.
THRESH = {"ela": (4.0, 8.0), "noise": (4.0, 8.0), "bglum": (2.5, 5.0), "inkdark": (3.5, 5.0)}
# ela & noise are both compression/ringing based -> one family; tone and ink darkness are independent
FAMILY = {"ela": "compression", "noise": "compression", "bglum": "paper_tone", "inkdark": "ink_darkness"}


def ela_map(bgr, q=90):
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.absdiff(bgr, dec).astype(np.float32).mean(axis=2)


def save_ela_heatmap(doc, path):
    ela = ela_map(read_rgb(doc))
    v = np.clip(ela / max(1e-6, np.percentile(ela, 99.5)) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.applyColorMap(v, cv2.COLORMAP_INFERNO))
    return path


def flatten_illumination(gray):
    """Divide out slowly varying page lighting (phone shadows / vignette) so tone and ink-darkness
    comparisons between fields are not dominated by the lighting gradient."""
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))   # removes thin text strokes
    bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), 60)
    return np.clip(gray.astype(np.float32) / np.maximum(bg, 1.0) * 240.0, 0, 255).astype(np.uint8)


def _metrics(gray, ela, resid, bbox, pad=8):
    h, w = gray.shape
    x0, y0, x1, y1 = bbox
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    g = gray[y0:y1, x0:x1]
    if g.size < 50:
        return None
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (g < t).astype(np.uint8)
    halo = cv2.dilate(ink, np.ones((7, 7), np.uint8))   # keep clear of JPEG ringing
    bg = halo == 0
    if bg.sum() < 300 or ink.sum() < 20:
        return None
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    stroke = 2.0 * float(np.percentile(dt[ink > 0], 75))
    return {
        "ela": float(ela[y0:y1, x0:x1][bg].mean()),
        "noise": float(resid[y0:y1, x0:x1][bg].std()),
        "bglum": float(g[bg].mean()),
        "inkdark": float(np.percentile(g[ink > 0], 10)),
        "stroke": stroke,
    }


def _robust_z(vals, i, rel, ab):
    arr = np.array(vals, dtype=float)
    others = np.delete(arr, i)                  # leave-one-out peers
    med = np.median(others)
    mad = 1.4826 * np.median(np.abs(others - med))
    scale = max(mad, rel * abs(med), ab)
    return (arr[i] - med) / scale, float(med)


def analyze(doc, audit, id_start=1):
    gray, bgr = read_gray(doc), read_rgb(doc)
    gray = flatten_illumination(gray)
    ela = ela_map(bgr)
    resid = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    rows = []
    for f in doc.fields:
        m = _metrics(gray, ela, resid, f.bbox)
        if m:
            rows.append((f, m))
    findings, table = [], []
    if len(rows) < 4:
        audit.log("forensics", doc_id=doc.doc_id, note="too few fields for peer comparison")
        return findings, table
    n = id_start
    for i, (f, m) in enumerate(rows):
        detail, flags = {}, []
        h = max(1, f.bbox[3] - f.bbox[1])
        grp = [j for j, (g, _) in enumerate(rows) if j == i or 0.75 <= max(1, g.bbox[3] - g.bbox[1]) / h <= 1.33]
        if len(grp) < 4:                      # header vs table text etc.: not enough comparable peers -> do not judge
            continue
        gi = grp.index(i)
        for name in FLOORS:
            z, med = _robust_z([rows[j][1][name] for j in grp], gi, *FLOORS[name])
            detail[name] = {"value": round(m[name], 3), "peer_median": round(med, 3), "z": round(z, 1)}
            if name in FLAGGING and abs(z) >= THRESH[name][0]:
                flags.append((name, abs(z)))
        table.append({"doc_id": doc.doc_id, "field": f.key, **{k: v["z"] for k, v in detail.items()}})
        fams = {FAMILY[k] for k, _ in flags}
        strong = [x for x in flags if x[1] >= THRESH[x[0]][1]]
        if len(fams) >= 2 or strong:
            maxz = max(z for _, z in flags)
            sev = "high" if (len(fams) >= 3 or len(flags) >= 3 or maxz >= 10) else "medium"
            conf = round(min(0.95, 0.40 + 0.15 * len(fams) + 0.03 * min(maxz, 10)), 2)
            names = ", ".join(f"{k} (z={detail[k]['z']:+.1f})" for k, _ in flags)
            findings.append(Finding(
                fid=f"V-{n:03d}", kind="visual",
                title=f"Possible alteration of '{f.label.strip(':')}' in {doc.doc_id}",
                severity=sev, confidence=conf, status="potential_manipulation",
                explanation=(f"The value region of '{f.label.strip(':')}' ({f.raw!r}) differs statistically from the "
                             f"other fields of the same page in: {names}. Such a pattern is typical of text that was "
                             f"edited or pasted after the page was first scanned/compressed. This is an indicator, "
                             f"not proof; verify against the original document."),
                refs=[{"doc_id": doc.doc_id, "field": f.key, "value": f.raw, "bbox": list(f.bbox)}],
                metrics=detail, method="peer-relative ELA/noise/background/stroke analysis"))
            n += 1
    audit.log("forensics", doc_id=doc.doc_id, fields_tested=len(rows), flagged=[x.refs[0]["field"] for x in findings])
    return findings, table
