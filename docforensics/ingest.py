import os, io, re, zipfile
import numpy as np
from PIL import Image, ImageOps
from .schema import Document
from .audit import sha256_file

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def load_pages(path, dpi=200):
    """Return list of PIL RGB images (PDF pages rendered, images loaded)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        import pymupdf
        doc = pymupdf.open(path)
        pages = []
        for k, p in enumerate(doc):
            if k >= MAX_PDF_PAGES:
                break
            pix = p.get_pixmap(dpi=dpi)
            pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        return pages
    if ext in IMG_EXT:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")     # phone photos carry an orientation flag
        if max(img.size) > MAX_SIDE:                                      # bound memory/time (e.g. 12 MP+ photos)
            r = MAX_SIDE / max(img.size)
            img = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)
        return [img]
    raise ValueError(f"Unsupported file type: {ext}")


MAX_ZIP_FILES, MAX_FILE_BYTES, MAX_TOTAL_BYTES = 200, 50 << 20, 300 << 20
MAX_SIDE, MAX_PDF_PAGES = 5000, 60          # longest image side in px; pages rendered per PDF
SUPPORTED = IMG_EXT | {".pdf"}


def expand_inputs(paths, workdir, audit):
    """Expand .zip inputs into supported files. Returns (paths, skipped[{name, reason}]).
    Safe unpacking: no extractall; entries are read and written under workdir with a sanitised
    flat name, so '../' or absolute paths cannot escape. Caps guard against zip bombs."""
    out, skipped = [], []
    for p in paths:
        if os.path.splitext(p)[1].lower() != ".zip":
            out.append(p)
            continue
        base = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(p))[0])
        dest = os.path.join(workdir, "_unzipped", base)
        os.makedirs(dest, exist_ok=True)
        try:
            zf = zipfile.ZipFile(p)
        except zipfile.BadZipFile:
            skipped.append({"name": os.path.basename(p), "reason": "not a valid zip file"})
            continue
        total, taken = 0, 0
        for info in zf.infolist():
            name = info.filename
            if info.is_dir():
                continue
            parts = [x for x in name.replace("\\", "/").split("/") if x]
            if name.startswith("/") or ".." in parts or re.match(r"^[A-Za-z]:", name):
                skipped.append({"name": name, "reason": "unsafe path"}); continue
            if parts and (parts[0] == "__MACOSX" or parts[-1].startswith(".")):
                continue                                        # OS metadata files
            ext = os.path.splitext(name)[1].lower()
            if ext not in SUPPORTED:
                skipped.append({"name": name, "reason": f"unsupported type {ext or '(none)'}"}); continue
            if info.flag_bits & 0x1:
                skipped.append({"name": name, "reason": "encrypted entry"}); continue
            if info.file_size > MAX_FILE_BYTES:
                skipped.append({"name": name, "reason": "file too large"}); continue
            if taken >= MAX_ZIP_FILES or total + info.file_size > MAX_TOTAL_BYTES:
                skipped.append({"name": name, "reason": "archive limit reached"}); continue
            flat = re.sub(r"[^A-Za-z0-9_.-]", "_", "__".join(parts))
            target = os.path.join(dest, flat)
            n_dup = 1
            while os.path.exists(target):                       # 'a/b.jpg' and 'a__b.jpg' must never overwrite each other
                stem, ext_ = os.path.splitext(flat)
                target = os.path.join(dest, f"{stem}__{n_dup}{ext_}")
                n_dup += 1
            with zf.open(info) as src, open(target, "wb") as dst:
                data = src.read(MAX_FILE_BYTES + 1)             # bounded read (header size can lie)
                if len(data) > MAX_FILE_BYTES:
                    skipped.append({"name": name, "reason": "file too large"}); continue
                dst.write(data)
            total += len(data); taken += 1
            out.append(target)
        audit.log("zip_expanded", source=os.path.basename(p), extracted=taken, skipped=len(skipped))
    for sk in skipped:
        audit.log("input_skipped", **sk)
    return out, skipped


def estimate_skew(gray, max_angle=8.0):
    """Page tilt in degrees via projection-profile search: the rotation that makes text rows
    sharpest (largest squared difference of the row-ink profile). Returns (angle, gain) where gain is
    the score improvement over 0 deg; apply only when both are significant."""
    import cv2
    h, w = gray.shape
    if w > 1000:
        gray = cv2.resize(gray, (1000, int(h * 1000 / w)), interpolation=cv2.INTER_AREA)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    h, w = bw.shape

    def score(a):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST)
        prof = r.sum(axis=1).astype(np.float64)
        return float(np.sum(np.diff(prof) ** 2))
    coarse = {a: score(a) for a in np.arange(-max_angle, max_angle + 1e-9, 0.5)}
    a0 = max(coarse, key=coarse.get)
    fine = {a: score(a) for a in np.arange(a0 - 0.5, a0 + 0.5 + 1e-9, 0.1)}
    best = max(fine, key=fine.get)
    return float(best), fine[best] / max(score(0.0), 1.0)


def deskew(img, audit=None, doc_id="", min_angle=0.3, min_gain=1.03):
    """Rotate a PIL page so text rows are horizontal. Untouched when tilt is negligible."""
    import cv2
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    angle, gain = estimate_skew(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    applied = abs(angle) >= min_angle and gain >= min_gain
    if applied:
        h, w = bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        bgr = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if audit:
        audit.log("deskew", doc_id=doc_id, estimated_angle=round(angle, 2), gain=round(gain, 3), applied=applied)
    return img, (round(angle, 2) if applied else 0.0)


def _pdf_born_digital(path, max_image_cover=0.35):
    """Per page: True when the page has a real text layer and raster images (logos, stamps) cover only a small part
    of it. A scanned/photographed PDF page is one big image, so its image coverage is high."""
    try:
        import pymupdf
        out = []
        for pg in pymupdf.open(path):
            area = max(1.0, pg.rect.width * pg.rect.height)
            cover = sum(max(0.0, (i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1])) for i in pg.get_image_info()) / area
            out.append(len(pg.get_text().strip()) >= 40 and cover < max_image_cover)
        return out
    except Exception:
        return []          # callers treat unknown as raster (pixel forensics run); logged by the caller if needed


def ingest(paths, workdir, audit):
    """Create Document objects; one per page. Stores normalised page PNG."""
    os.makedirs(workdir, exist_ok=True)
    docs, used_ids = [], set()
    for p in paths:
        digest = sha256_file(p)
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            pages = load_pages(p)
        except Exception as e:                                  # unsupported / corrupt / encrypted
            audit.log("input_skipped", name=os.path.basename(p), reason=f"{type(e).__name__}: {e}"[:160])
            continue
        bd = _pdf_born_digital(p) if os.path.splitext(p)[1].lower() == ".pdf" else []
        file_meta = None
        for i, img in enumerate(pages):
            doc_id = base if len(pages) == 1 else f"{base}_p{i+1}"
            if doc_id in used_ids:                              # same stem from different files (scan.jpg + scan.png)
                doc_id = f"{doc_id}-{digest[:6]}"
            k_ = 2
            while doc_id in used_ids:
                doc_id, k_ = f"{doc_id.split('~')[0]}~{k_}", k_ + 1
            used_ids.add(doc_id)
            out = os.path.join(workdir, f"{doc_id}.png")
            born = bool(bd[i]) if i < len(bd) else False
            if born:
                skew = 0.0                                  # generated page: never rotate it (text-layer coordinates must match the picture)
            else:
                img, skew = deskew(img, audit, doc_id)      # normalised page used by every later stage
            img.save(out)
            d = Document(doc_id=doc_id, path=p, sha256=digest, image_path=out,
                         width=img.width, height=img.height, skew_deg=skew, born_digital=born)
            if born:
                from .pdf_layer import read_page, read_meta
                lay = read_page(p, i) if os.environ.get("DOCFOR_PDF_TEXT", "auto") != "off" else None     # off: read born-digital pages by OCR like scans
                if lay and lay["words"]:                    # the text layer IS the text: exact, no OCR needed
                    d.words = lay.pop("words")
                    d.pdf = lay
                    file_meta = file_meta or read_meta(p)
                    d.pdf["meta"] = file_meta
                    d.facts["ocr_engine"] = "pdf-text-layer"
                    d.facts["ocr_mean_conf"] = 1.0
                elif lay is not None:
                    d.born_digital = False                  # text layer unreadable: treat as an image and OCR it
            docs.append(d)
            audit.log("ingest", doc_id=doc_id, source=os.path.basename(p), sha256=digest,
                      size=[img.width, img.height])
    return docs


def read_gray(doc):
    import cv2
    return cv2.imread(doc.image_path, cv2.IMREAD_GRAYSCALE)


def read_rgb(doc):
    import cv2
    return cv2.imread(doc.image_path, cv2.IMREAD_COLOR)
