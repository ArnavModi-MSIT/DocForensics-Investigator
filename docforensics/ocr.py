import os, shutil
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


_locate_tesseract()


def run_ocr(doc, audit):
    img = Image.open(doc.image_path)
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT,
                                  config="--psm 11")
    words = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if not t:
            continue
        c = float(d["conf"][i])
        if c < 0:
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        words.append(Word(t, (x, y, x + w, y + h), c / 100.0))
    doc.words = words
    audit.log("ocr", doc_id=doc.doc_id, engine="tesseract", n_words=len(words),
              mean_conf=round(sum(w.conf for w in words) / max(1, len(words)), 3))
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
