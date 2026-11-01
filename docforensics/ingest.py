import os, io, re, zipfile
import numpy as np
from PIL import Image
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
        for p in doc:
            pix = p.get_pixmap(dpi=dpi)
            pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        return pages
    if ext in IMG_EXT:
        return [Image.open(path).convert("RGB")]
    raise ValueError(f"Unsupported file type: {ext}")


MAX_ZIP_FILES, MAX_FILE_BYTES, MAX_TOTAL_BYTES = 200, 50 << 20, 300 << 20
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
            with zf.open(info) as src, open(target, "wb") as dst:
                data = src.read(MAX_FILE_BYTES + 1)             # bounded read (header size can lie)
                if len(data) > MAX_FILE_BYTES:
                    skipped.append({"name": name, "reason": "file too large"}); continue
                dst.write(data)
            total += len(data); taken += 1
            out.append(target)
        audit.log("zip_expanded", source=p, extracted=taken, skipped=len(skipped))
    for sk in skipped:
        audit.log("input_skipped", **sk)
    return out, skipped


def ingest(paths, workdir, audit):
    """Create Document objects; one per page. Stores normalised page PNG."""
    os.makedirs(workdir, exist_ok=True)
    docs = []
    for p in paths:
        digest = sha256_file(p)
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            pages = load_pages(p)
        except Exception as e:                                  # unsupported / corrupt / encrypted
            audit.log("input_skipped", name=os.path.basename(p), reason=f"{type(e).__name__}: {e}"[:160])
            continue
        for i, img in enumerate(pages):
            doc_id = base if len(pages) == 1 else f"{base}_p{i+1}"
            out = os.path.join(workdir, f"{doc_id}.png")
            img.save(out)
            d = Document(doc_id=doc_id, path=p, sha256=digest, image_path=out,
                         width=img.width, height=img.height)
            docs.append(d)
            audit.log("ingest", doc_id=doc_id, source=p, sha256=digest,
                      size=[img.width, img.height])
    return docs


def read_gray(doc):
    import cv2
    return cv2.imread(doc.image_path, cv2.IMREAD_GRAYSCALE)


def read_rgb(doc):
    import cv2
    return cv2.imread(doc.image_path, cv2.IMREAD_COLOR)
