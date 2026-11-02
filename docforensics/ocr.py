import os, re, shutil
import cv2
import numpy as np
import pytesseract
from PIL import Image
from .schema import Word


def _locate_tesseract():
    """TESSERACT_CMD env var > PATH > common Windows install locations."""
    cands = [os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"),
             r"C:\Program Files\Tesseract-OCR\tesseract.exe",
             r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
             os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
             os.path.expanduser(r"~\Downloads\tesseract-5.5.4\tesseract.exe")]
    for c in cands:
        if c and os.path.isfile(c):
            pytesseract.pytesseract.tesseract_cmd = c
            return c
    raise RuntimeError("Tesseract executable not found. Install the Windows build from "
                       "https://github.com/UB-Mannheim/tesseract/wiki or set TESSERACT_CMD to tesseract.exe")


_ready = {"ok": False}


def ensure_tesseract():
    if not _ready["ok"]:
        _locate_tesseract()
        _ready["ok"] = True


# ---------------------------------------------------------------- engines
# Tesseract is the baseline. When a page reads poorly (busy coloured backgrounds, photos) we retry with classic
# preprocessing variants and, if installed, a local RapidOCR engine (ONNX, no network, no API). The best-scoring
# reading wins per page. Everything is optional except Tesseract itself.
#   DOCFOR_OCR = auto (default) | tesseract (no retries, no RapidOCR) | rapid (RapidOCR first, Tesseract fallback)
#   DOCFOR_OCR_RETRY_CONF = mean word confidence below which retries run (default 0.60)
RETRY_CONF = float(os.environ.get("DOCFOR_OCR_RETRY_CONF", "0.60"))
_rapid = {"engine": None, "tried": False}


def _mode():
    return os.environ.get("DOCFOR_OCR", "auto").strip().lower()


def _get_rapid():
    """Lazy, optional RapidOCR. Returns a function  bgr_image -> [(box, text, confidence), ...]  or None.
    Prefers the current `rapidocr` package (any Python >= 3.8, models bundled; needs `onnxruntime`), falls back to the
    legacy `rapidocr_onnxruntime` (Python < 3.13)."""
    if not _rapid["tried"]:
        _rapid["tried"] = True
        try:
            from rapidocr import RapidOCR
            try:
                eng = RapidOCR(params={"Global.log_level": "warning"})
            except Exception:
                eng = RapidOCR()

            def call(img):
                r = eng(img)
                if r is None or r.boxes is None or r.txts is None:
                    return []
                return [(b.tolist() if hasattr(b, "tolist") else b, t, float(c)) for b, t, c in zip(r.boxes, r.txts, r.scores)]
            _rapid["engine"] = call
        except Exception:
            try:
                from rapidocr_onnxruntime import RapidOCR as Legacy
                eng = Legacy()
                _rapid["engine"] = lambda img: [(b, t, float(c)) for b, t, c in (eng(img)[0] or [])]
            except Exception:
                _rapid["engine"] = None
    return _rapid["engine"]


def rapidocr_available():
    return _get_rapid() is not None


def _tess_words(arr, scale=1.0):
    """Tesseract on a numpy image; boxes mapped back to original coordinates by dividing by `scale`."""
    d = pytesseract.image_to_data(Image.fromarray(arr), output_type=pytesseract.Output.DICT,
                                  config="--psm 11", timeout=120)
    out = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        c = float(d["conf"][i])
        if not t or c < 0:
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        out.append(Word(t, (int(x / scale), int(y / scale), int((x + w) / scale), int((y + h) / scale)), c / 100.0))
    return out


_glue = {"phrases": None}


_vocab = {"words": None}
_EXTRA_WORDS = ("of the and for to no an india government department income tax election commission identity card elector "
                "photo number account permanent father mother name date birth unique identification authority bank "
                "statement salary slip pay month net gross address issue sex male female")


def _segment(core):
    """Split an all-letter token that is exactly covered by 2+ known document words ('GOVERNMENTOFINDIA' ->
    Government of India). Vocabulary = our own labels, type keywords and ID cues, so ordinary names never split."""
    if len(core) < 7 or not core.isalpha():
        return None
    if _vocab["words"] is None:
        from .extract import LABELS, DOC_TYPES                    # lazy: extract imports this module
        from .id_patterns import STRONG_CUES
        ws = set(_EXTRA_WORDS.split())
        for group in (LABELS.values(), DOC_TYPES.values(), STRONG_CUES.values()):
            for cands in group:
                for c in cands:
                    ws.update(re.sub(r"[^a-z ]", "", c.lower()).split())
        _vocab["words"] = {w for w in ws if len(w) >= 2}
    V, low = _vocab["words"], core.lower()
    if low in V:
        return None
    best = [None] * (len(low) + 1)                                # best[i]: segmentation of low[:i] with fewest pieces
    best[0] = []
    for i in range(1, len(low) + 1):
        for j in range(max(0, i - 14), i):
            if best[j] is not None and low[j:i] in V and (best[i] is None or len(best[j]) + 1 < len(best[i])):
                best[i] = best[j] + [(j, i)]
    if best[-1] is None or len(best[-1]) < 2:
        return None
    return [core[a:b] for a, b in best[-1]]


def _respace(tok, first=True):
    """RapidOCR often drops spaces ('SexMale', 'DateofBirth:'). Split camelCase joins, and re-split a token that is a
    known multi-word label written without spaces (fuzzy, so 'Date6fBirth:' still works). Returns a list of tokens."""
    if _glue["phrases"] is None:
        from .extract import LABELS                               # lazy: extract imports this module
        _glue["phrases"] = [c for cands in LABELS.values() for c in cands if " " in c]
    from rapidfuzz import fuzz
    core, tail = re.match(r"^(.*?)([:.\-]*)$", tok).groups()
    letters = re.sub(r"[^a-z]", "", core.lower())
    if len(letters) >= 6:
        for ph in _glue["phrases"]:
            if fuzz.ratio(letters, ph.replace(" ", "").replace("'", "")) >= 88:
                words = ph.split()
                words[-1] += tail
                return words
    seg = _segment(core)
    if seg:
        seg[-1] += tail
        return seg
    if first:                                                 # only line-leading tokens: keeps names like 'MacArthur' intact
        return re.sub(r"([a-z]{2,})(?=[A-Z][a-z])", r"\1 ", tok).split() or [tok]
    return [tok]


def _rapid_words(arr_bgr):
    """RapidOCR returns text lines; split each into words with boxes shared in proportion to character count."""
    eng = _get_rapid()
    if eng is None:
        return []
    out = []
    for box, text, conf in eng(arr_bgr):
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        text = re.sub(r"([:;])(?=[A-Za-z0-9])", r"\1 ", text)         # "Name:Rahul" -> "Name: Rahul"
        toks = [p for i, t in enumerate(text.split()) for p in _respace(t, first=(i == 0))]
        n = sum(len(t) for t in toks) + max(0, len(toks) - 1)
        if not toks or n == 0 or x1 <= x0:
            continue
        cx = x0
        for t in toks:
            w = (x1 - x0) * len(t) / n
            out.append(Word(t, (int(cx), int(y0), int(cx + w), int(y1)), float(conf)))
            cx += w + (x1 - x0) / n
    return out


def _variants(bgr):
    """Cheap classic preprocessing variants: (name, image for Tesseract, scale)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    up = 2.0 if max(h, w) < 2600 else 1.0
    big = cv2.resize(gray, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up != 1.0 else gray
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(big)
    yield "up_clahe", clahe, up
    k = max(15, (int(min(big.shape) / 20) | 1))
    yield "adaptive", cv2.adaptiveThreshold(cv2.GaussianBlur(big, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, k, 12), up
    mn = bgr.min(axis=2)                                    # min-RGB suppresses coloured (tricolour/guilloche) backgrounds
    mn = cv2.resize(mn, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up != 1.0 else mn
    yield "minrgb", cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(mn), up


def _plausible(w):
    """A token that looks like real text: >=3 letters with a vowel, or a >=3-char token containing digits."""
    t = re.sub(r"[^A-Za-z0-9]", "", w.text)
    if len(t) < 3 or w.conf < 0.6:
        return False
    if any(c.isdigit() for c in t):
        return True
    return t.isalpha() and bool(re.search(r"[AEIOUYaeiouy]", t))


def _score(words):
    """Quality of a reading: confidence mass of plausible words minus a penalty for junk volume, so a noisy
    variant that emits hundreds of garbage tokens cannot beat a clean reading with fewer words."""
    good = sum(w.conf for w in words if _plausible(w))
    junk = sum(1 for w in words if not _plausible(w) and w.conf < 0.5)
    return good - 0.15 * junk


def _mean(words):
    return sum(w.conf for w in words) / max(1, len(words))


def _poor(words):
    return len(words) < 8 or _mean(words) < RETRY_CONF


def run_ocr(doc, audit):
    ensure_tesseract()
    mode = _mode()
    bgr = cv2.imread(doc.image_path, cv2.IMREAD_COLOR)
    if bgr is None:                                          # unusual format: let PIL decode it
        bgr = cv2.cvtColor(np.array(Image.open(doc.image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    tried = []                                               # (engine, words) audit trail

    def attempt(name, fn):
        try:
            ws = fn()
        except Exception as e:                               # an engine failure must never kill the pipeline
            audit.log("ocr_variant_failed", doc_id=doc.doc_id, engine=name, error=type(e).__name__)
            return []
        tried.append((name, ws))
        return ws

    def best():
        return max(tried, key=lambda t: _score(t[1]))[1] if tried else []

    if mode == "rapid" and rapidocr_available():
        attempt("rapidocr", lambda: _rapid_words(bgr))
    if not tried or _poor(tried[0][1]):
        attempt("tesseract", lambda: _tess_words(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
    if mode not in ("tesseract", "rapid") and _poor(best()):
        if rapidocr_available() and not any(n == "rapidocr" for n, _ in tried):   # fast and usually the best fix
            attempt("rapidocr", lambda: _rapid_words(bgr))
        if _poor(best()):                                    # still poor (or RapidOCR not installed): classic variants
            for name, img, sc in _variants(bgr):
                attempt("tesseract:" + name, lambda img=img, sc=sc: _tess_words(img, sc))

    name, words = max(tried, key=lambda t: _score(t[1])) if tried else ("none", [])
    doc.words = words
    audit.log("ocr", doc_id=doc.doc_id, engine=name, n_words=len(words), mean_conf=round(_mean(words), 3),
              candidates={n: [len(w), round(_score(w), 2)] for n, w in tried})
    doc.facts["ocr_engine"] = name
    doc.facts["ocr_mean_conf"] = round(_mean(words), 3)
    return words


def group_lines(words):
    """Cluster words into visual lines by vertical centre (robust to wide gaps)."""
    if not words:
        return []
    hs = sorted(w.bbox[3] - w.bbox[1] for w in words)
    med_h = hs[len(hs) // 2]
    ws = sorted(words, key=lambda w: (w.bbox[1] + w.bbox[3]) / 2)
    lines, cur, cy = [], [], None
    for w in ws:
        c = (w.bbox[1] + w.bbox[3]) / 2
        if cy is None or abs(c - cy) <= 0.6 * med_h:
            cur.append(w)
            cy = sum((x.bbox[1] + x.bbox[3]) / 2 for x in cur) / len(cur)
        else:
            lines.append(sorted(cur, key=lambda x: x.bbox[0]))
            cur, cy = [w], c
    if cur:
        lines.append(sorted(cur, key=lambda x: x.bbox[0]))
    return lines
