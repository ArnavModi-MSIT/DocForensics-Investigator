"""PDF layer access for born-digital pages.

A generated PDF already carries its text: reading that text layer is exact (no OCR misreads, confidence 1.0) and it also
exposes things a picture of the page hides: the font of every piece of text, the rectangles drawn behind or over it, text
that is invisible or covered, and how the file was saved. This module only READS the PDF; the checks built on top of it live in
pdf_forensics.py.
"""
import re
from .schema import Word

PDF_DPI = 200                       # must equal the dpi used by ingest.load_pages
SCALE = PDF_DPI / 72.0


def _px(b):
    return tuple(round(float(v) * SCALE, 1) for v in b)


WORD_H = 0.66      # a PDF word box spans the font's ascent + descent (about 1.2 x the font size); OCR boxes hug the glyphs (about 0.75 x). Every
                   # threshold downstream (line grouping, cell gaps, table headers) is calibrated on OCR-like boxes, so the height is scaled to match


def _word_px(b):
    x0, y0, x1, y1 = (float(v) * SCALE for v in b)
    cy, h = (y0 + y1) / 2, (y1 - y0) * WORD_H
    return (round(x0, 1), round(cy - h / 2, 1), round(x1, 1), round(cy + h / 2, 1))


def _rgb(c):
    """pymupdf colour (tuple of 0-1 floats, or int) -> (r, g, b) 0-1 floats or None."""
    if c is None:
        return None
    if isinstance(c, int):
        return (((c >> 16) & 255) / 255.0, ((c >> 8) & 255) / 255.0, (c & 255) / 255.0)
    c = tuple(float(x) for x in c)
    return c if len(c) == 3 else ((c[0],) * 3 if len(c) == 1 else None)


def _rotated_lines(pg, min_angle=0.08):
    """[(bbox_pt, text)] of text lines whose direction is not horizontal (|sin| > min_angle)."""
    found = []
    try:
        for blk in pg.get_text("dict").get("blocks", ()):
            for ln in blk.get("lines", ()):
                dx, dy = ln.get("dir", (1.0, 0.0))
                if abs(dy) > min_angle:
                    txt = " ".join(sp.get("text", "") for sp in ln.get("spans", ())).strip()
                    if txt:
                        found.append((tuple(float(v) for v in ln["bbox"]), txt))
    except Exception:
        pass
    return found


def _in_rotated(box, text, rotated, tol=3.0):
    """True when the word box sits inside a rotated line's box and its text belongs to that line."""
    for bb, txt in rotated:
        if text in txt and box[0] >= bb[0] - tol and box[1] >= bb[1] - tol and box[2] <= bb[2] + tol and box[3] <= bb[3] + tol:
            return True
    return False


def read_page(path, index):
    """Everything the PDF says about page `index`: words (as schema.Word in page-image pixels), spans (text with font, size,
    colour, draw order), drawings (filled / stroked shapes with draw order) and images. Returns None when the page cannot be read."""
    try:
        import pymupdf
        pdf = pymupdf.open(path)
        pg = pdf[index]
    except Exception:
        return None
    out = {"words": [], "spans": [], "drawings": [], "images": [], "page_pt": [pg.rect.width, pg.rect.height]}
    rotated = _rotated_lines(pg)                           # diagonal / vertical text (watermarks such as "SAMPLE" or "COPY") is not part of the reading order
    out["rotated_text"] = [t for _, t in rotated]
    try:
        for x0, y0, x1, y1, t, *_ in pg.get_text("words"):
            t = t.strip()
            if t and not _in_rotated((x0, y0, x1, y1), t, rotated):
                out["words"].append(Word(t, _word_px((x0, y0, x1, y1)), 1.0))
    except Exception:
        return None
    try:                                                   # text trace: every text run in DRAW order (seqno shared with drawings)
        for s in pg.get_texttrace():
            txt = "".join(chr(ch[0]) for ch in s.get("chars", ()) if ch and ch[0] > 0).strip()
            if not txt:
                continue
            out["spans"].append({
                "text": txt, "font": str(s.get("font", "")), "size": round(float(s.get("size", 0)), 2),
                "bbox": _px(s["bbox"]), "bbox_pt": [round(float(v), 1) for v in s["bbox"]],
                "color": _rgb(s.get("color")), "opacity": float(s.get("opacity", 1.0)),
                "seqno": int(s.get("seqno", 0)), "render": int(s.get("type", 0)),         # type 3 = invisible text
                "bold": bool(int(s.get("flags", 0)) & 16) if "flags" in s else ("bold" in str(s.get("font", "")).lower()),
            })
    except Exception:
        pass
    try:
        for d in pg.get_drawings():
            r = d.get("rect")
            if r is None:
                continue
            out["drawings"].append({
                "rect": _px(r), "rect_pt": [round(float(v), 1) for v in r], "fill": _rgb(d.get("fill")), "stroke": _rgb(d.get("color")),
                "width": d.get("width"), "seqno": int(d.get("seqno", 0)), "type": d.get("type", ""),
                "opacity": float(d.get("fill_opacity") if d.get("fill_opacity") is not None else 1.0)})
    except Exception:
        pass
    try:
        for im in pg.get_image_info():
            out["images"].append({"rect": _px(im["bbox"])})
    except Exception:
        pass
    return out


def read_meta(path):
    """File-level facts: creator tools, dates, and how many times the file was saved (incremental updates leave several %%EOF markers)."""
    meta = {"producer": None, "creator": None, "created": None, "modified": None, "revisions": 1, "encrypted": False, "pages": 0}
    try:
        import pymupdf
        pdf = pymupdf.open(path)
        m = pdf.metadata or {}
        meta.update(producer=m.get("producer") or None, creator=m.get("creator") or None, created=m.get("creationDate") or None,
                    modified=m.get("modDate") or None, encrypted=bool(pdf.is_encrypted), pages=len(pdf))
        meta["title"] = m.get("title") or None
        meta["xmp"] = parse_xmp(pdf.get_xml_metadata())
    except Exception:
        pass
    try:
        with open(path, "rb") as fh:
            data = fh.read(60 << 20)
        meta["revisions"] = max(1, len(re.findall(rb"%%EOF", data)))
        meta["signed"] = b"/ByteRange" in data                          # a digital signature is present (signing legitimately adds a revision)
    except Exception:
        pass
    return meta


def parse_xmp(xml):
    """The few XMP properties that duplicate the Info dictionary: creator tool, producer, create / modify dates (None when absent)."""
    out = {}
    if not xml:
        return out
    for key, tags in (("creator", ("xmp:CreatorTool",)), ("producer", ("pdf:Producer",)), ("created", ("xmp:CreateDate",)), ("modified", ("xmp:ModifyDate",))):
        for t in tags:
            m = re.search(rf"<{t}>([^<]+)</{t}>", xml) or re.search(rf'{t}="([^"]+)"', xml)
            if m:
                out[key] = m.group(1).strip()
                break
    return out


def to_utc(s):
    """A PDF Info date ('D:20261103085713+05\'30\'') or an XMP / ISO date -> (naive UTC datetime, had_timezone) or (None, False)."""
    from datetime import datetime, timedelta, timezone
    s = (s or "").strip()
    m = re.match(r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?\s*(?:([+-])(\d{2})'?(\d{2})?'?|Z)?", s)
    try:
        if m:
            parts = [int(g) if g else d for g, d in zip(m.groups()[:6], (0, 1, 1, 0, 0, 0))]
            dt = datetime(*parts)
            if m.group(7):
                off = timedelta(hours=int(m.group(8)), minutes=int(m.group(9) or 0))
                return (dt - off if m.group(7) == "+" else dt + off), True
            return dt, s.endswith("Z")
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo:
            return dt.astimezone(timezone.utc).replace(tzinfo=None), True
        return dt, False
    except ValueError:
        return None, False


def parse_pdf_date(s):
    """'D:20261103085713+00'00'' -> datetime (naive UTC) or None."""
    from datetime import datetime
    m = re.match(r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?", s or "")
    if not m:
        return None
    parts = [int(g) if g else d for g, d in zip(m.groups(), (0, 1, 1, 0, 0, 0))]
    try:
        return datetime(*parts)
    except ValueError:
        return None


def widen(box):
    """Inverse of the word-box scaling: a box in analysis geometry -> the PDF's own word geometry (ascent to descent), which is what a
    reader sees as the text's extent and what other tools report for the same text."""
    x0, y0, x1, y1 = box
    cy, h = (y0 + y1) / 2, (y1 - y0) / WORD_H
    return [round(x0, 1), round(cy - h / 2, 1), round(x1, 1), round(cy + h / 2, 1)]


def restore_pdf_geometry(docs, findings):
    """After every analysis step: boxes of fields and of finding references on born-digital pages are reported in the PDF's own word
    geometry. References that already carry exact geometry (rectangles read from the drawing records) are left alone."""
    bd = {d.doc_id for d in docs if d.born_digital and d.pdf}
    for f in findings:
        for r in f.refs:
            if r.get("bbox") and r.get("doc_id") in bd and not r.get("exact"):
                r["bbox"] = widen(r["bbox"])
    for d in docs:
        if d.doc_id in bd:
            for fld in d.fields:
                fld.bbox = tuple(widen(fld.bbox))
