"""Visual forensics part 2: typography / alignment / spacing, JPEG block grid, photo region, signature region.
Everything is fabricated from drawing coordinates (no OCR), so the tests run in seconds."""
import io, os, sys
import numpy as np
import pytest
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from docforensics import forensics_ext as fe
from docforensics.schema import Document, Word, Field
from visual_docs import id_card, signature_form, jpeg


class Audit:
    def log(self, *a, **k):
        pass


def _png(im, tmp_path, name):
    p = tmp_path / name
    im.save(p)
    return str(p)


def _usable(doc, path):
    doc.image_path, doc.skew_deg, doc.born_digital = path, 0.0, False
    return doc


# ---- typography, alignment, spacing ----------------------------------------------------------------------
def form_doc(n=6, value_h=None, shift=None, slope=0.0, gap_shift=None, engine="tesseract"):
    """n label:value rows, label at x=60, value at x=430 (+slope*y); `value_h[i]` overrides a value's text height."""
    d = Document(doc_id="f_p1", path="f.png", sha256="0", image_path="")
    d.doc_type, d.width, d.height = "application_form", 1240, 1754
    d.facts["ocr_engine"] = engine
    for i in range(n):
        y = 200 + i * 85 + ((gap_shift or {}).get(i, 0))
        h = (value_h or {}).get(i, 28)
        x = 430 + slope * y + ((shift or {}).get(i, 0))
        cy = y + 14
        lab = Word(f"Label{i}:", (60, y, 360, y + 28), 0.95)
        val = Word(f"Value{i}", (int(x), int(cy - h / 2), int(x) + 16 * 7 * h // 28, int(cy + h / 2)), 0.95)
        d.words += [lab, val]
        d.fields.append(Field(f"k{i}", f"Label{i}:", val.text, val.text, val.bbox, 0.95, "fuzzy"))
    return d


def test_gross_text_height_change_is_flagged_but_ordinary_fields_are_not():
    d = form_doc(value_h={2: 64})
    f, table = fe.analyze_typography(d, Audit())
    assert [x.refs[0]["field"] for x in f] == ["k2"] and f[0].status == "potential_manipulation" and f[0].severity == "low"
    assert not fe.analyze_typography(form_doc(), Audit())[0]


def test_rapidocr_boxes_are_not_used_for_box_signals():
    d = form_doc(value_h={2: 64}, engine="rapidocr")
    assert fe.analyze_typography(d, Audit())[0] == []                     # estimated word boxes: height / width / baseline skipped


def test_value_out_of_its_column_is_flagged_even_on_a_tilted_page():
    f = fe.analyze_typography(form_doc(shift={3: 130}), Audit())[0]
    assert any("misaligned" in x.title and x.refs[0]["field"] == "k3" for x in f)
    tilted = fe.analyze_typography(form_doc(slope=0.04), Audit())[0]      # whole column slanted (phone photo): not an anomaly
    assert not any("misaligned" in x.title for x in tilted)


def test_single_line_shifted_between_its_neighbours_is_flagged():
    f = fe.analyze_typography(form_doc(gap_shift={3: 30}), Audit())[0]
    assert any("line spacing" in x.title and x.refs[0]["field"] == "k3" for x in f)
    assert not any("line spacing" in x.title for x in fe.analyze_typography(form_doc(), Audit())[0])


def test_too_few_fields_is_reported_not_guessed():
    d = form_doc(n=3)
    assert fe.analyze_typography(d, Audit()) == ([], []) and d.facts["typography"].startswith("not run")


# ---- JPEG block grid -----------------------------------------------------------------------------------
def _page(tmp_path, q=60):
    im, doc = signature_form("genuine")
    return np.asarray(jpeg(im.convert("L").convert("RGB"), q).convert("L")), doc


def test_clean_jpeg_page_has_one_dominant_grid_and_no_region(tmp_path):
    page, doc = _page(tmp_path, 75)
    f, regs, note = fe.analyze_grid(_usable(doc, "x"), page, Audit())
    assert note.startswith("run on") and f == [] and regs == []


def test_pasted_content_with_its_own_grid_is_found(tmp_path):
    page, doc = _page(tmp_path, 60)
    buf = io.BytesIO(); Image.fromarray(np.ascontiguousarray(page[203:333, 103:503])).save(buf, "JPEG", quality=50); buf.seek(0)
    patch = np.asarray(Image.open(buf).convert("L"))
    forged = page.copy(); forged[203:203 + patch.shape[0], 103:103 + patch.shape[1]] = patch      # misaligned with the page's 8x8 grid
    f, regs, note = fe.analyze_grid(_usable(doc, "x"), forged, Audit())
    assert f and any(r["bbox"][0] <= 160 and r["bbox"][2] >= 400 for r in regs)


def test_resampled_or_rotated_or_pdf_pages_are_skipped_with_a_reason():
    d = Document(doc_id="p", path="p", sha256="0", image_path="")
    d.skew_deg = 1.2
    assert "rotated" in fe.analyze_grid(d, np.zeros((200, 200), np.uint8), Audit())[2]
    d.skew_deg, d.born_digital = 0.0, True
    assert "PDF" in fe.analyze_grid(d, np.zeros((200, 200), np.uint8), Audit())[2]
    d.born_digital = False
    assert "dominant" in fe.analyze_grid(d, (np.random.default_rng(0).random((400, 400)) * 255).astype(np.uint8), Audit())[2] or \
           "too little" in fe.analyze_grid(d, (np.random.default_rng(0).random((400, 400)) * 255).astype(np.uint8), Audit())[2]


# ---- photo region ---------------------------------------------------------------------------------------
def test_photo_is_located_and_the_qr_code_is_not_taken_for_one(tmp_path):
    im, doc = id_card()
    doc = _usable(doc, _png(im, tmp_path, "c.png"))
    gray = fe.flatten_illumination(fe.read_gray(doc))
    bb = fe.find_photo(doc, gray)
    assert bb and bb[0] < 60 and bb[1] < 130 and bb[2] < 300           # the portrait at the left, not the QR block at x>=740


def test_genuine_photo_matches_the_page_grid_but_a_pasted_one_does_not(tmp_path):
    for forged in (False, True):
        im, doc = id_card(forged_photo=forged)
        doc = _usable(doc, _png(im, tmp_path, f"c{forged}.png"))
        f, regs, note = fe.analyze_photo(doc, fe.read_gray(doc), Audit())
        assert regs and regs[0]["kind"] == "photo"
        assert bool(f) == forged, note


def test_photo_check_only_for_id_cards():
    d = Document(doc_id="p", path="p", sha256="0", image_path="")
    d.doc_type = "bank_statement"
    assert fe.analyze_photo(d, np.zeros((10, 10), np.uint8), Audit())[2] == "not an ID card"


# ---- signature -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind,expected", [("genuine", []), ("none", ["No signature found"]), ("overlap", ["Signature overlaps printed text"]),
                                           ("pasted", ["Signature region differs from the surrounding paper"])])
def test_signature_scenarios(tmp_path, kind, expected):
    im, doc = signature_form(kind)
    doc = _usable(doc, _png(im, tmp_path, f"s_{kind}.png"))
    f, regs, note = fe.analyze_signature(doc, fe.read_gray(doc), Audit())
    titles = [x.title for x in f]
    assert all(any(e in t for t in titles) for e in expected) and (expected or not titles), titles
    if kind == "genuine":
        assert regs[0]["found"] and regs[0]["placement"] == "above the label"


def test_no_signature_label_means_no_signature_check():
    d = form_doc()
    assert fe.analyze_signature(_usable(d, "x"), np.zeros((1754, 1240), np.uint8), Audit())[2] == "no signature label on the page"


def test_driver_never_raises_and_records_what_ran(tmp_path):
    im, doc = signature_form("none")
    doc = _usable(doc, _png(im, tmp_path, "d.png"))
    doc.facts["ocr_engine"] = "tesseract"
    f = fe.analyze_ext(doc, Audit())
    assert any("No signature found" in x.title for x in f)
    assert doc.facts["typography"] and doc.facts["grid"] and doc.facts["regions"]
    bad = Document(doc_id="b", path="b", sha256="0", image_path="/does/not/exist.png")
    fe.analyze_ext(bad, Audit())                                           # must not raise
    assert "regions_error" in bad.facts


# ---- unsupported pages (e.g. a certificate with a signature and no 'Signature' label) -------------------------
def test_unlabelled_signature_is_located_and_seal_qr_border_are_not(tmp_path):
    from visual_docs import certificate
    im, doc = certificate(signature=True)
    doc = _usable(doc, _png(im, tmp_path, "c.png"))
    f, regs, note = fe.analyze_signature(doc, fe.read_gray(doc), Audit(), allow_unlabelled=True)
    assert len(regs) == 1 and regs[0]["labelled"] is False and f == []
    x0, y0, x1, y1 = regs[0]["bbox"]
    assert 600 <= x0 and x1 <= 1060 and 780 <= y0 and y1 <= 910                       # on the handwriting, not the seal (x>=1180) or the QR (x>=1450)


def test_no_mark_means_no_unlabelled_signature(tmp_path):
    from visual_docs import certificate
    im, doc = certificate(signature=False)
    doc = _usable(doc, _png(im, tmp_path, "n.png"))
    f, regs, note = fe.analyze_signature(doc, fe.read_gray(doc), Audit(), allow_unlabelled=True)
    assert regs == [] and "no handwriting-like mark" in note


def test_labelled_certificate_uses_the_labelled_path(tmp_path):
    from visual_docs import certificate
    im, doc = certificate(signature=True, labelled=True)
    doc = _usable(doc, _png(im, tmp_path, "l.png"))
    f, regs, note = fe.analyze_signature(doc, fe.read_gray(doc), Audit(), allow_unlabelled=True)
    assert regs and regs[0]["found"] and regs[0].get("labelled", True) is not False


def test_signature_only_driver_records_the_result_for_unsupported_pages(tmp_path):
    from visual_docs import certificate
    im, doc = certificate()
    doc = _usable(doc, _png(im, tmp_path, "s.png"))
    fe.analyze_signature_only(doc, Audit())
    assert doc.facts["regions"] and "handwriting-like" in doc.facts["signature"]


def test_reason_line_leads_with_the_semantic_category():
    from docforensics.extract import lead_with_category
    d = Document(doc_id="x", path="x", sha256="0", image_path="")
    d.class_info = {"decision": "only 0 of 6 expected 'application_form' fields found (need 3)",
                    "category": {"key": "other:certificate", "label": "Certificate", "confidence": 0.76}}
    lead_with_category(d)
    assert d.class_info["decision"].startswith("most similar to 'Certificate' (semantic confidence 0.76)") and "0 of 6" in d.class_info["decision"]
    d2 = Document(doc_id="y", path="y", sha256="0", image_path="")                    # a supported page is left alone
    d2.doc_type, d2.class_info = "id_card", {"decision": "accepted as id_card", "category": {"key": "other:certificate", "label": "Certificate", "confidence": 0.9}}
    lead_with_category(d2)
    assert d2.class_info["decision"] == "accepted as id_card"


def test_bold_versus_regular_never_flags_and_a_long_label_is_not_misalignment():
    """Real documents set totals and names in bold: the pixel signals are reported but never flag."""
    import cv2
    d = form_doc()
    img = np.full((1754, 1240), 245, np.uint8)
    for i, f in enumerate(d.fields):                                       # field 2 drawn much heavier than the rest
        x0, y0, x1, y1 = f.bbox
        cv2.putText(img, "Value%d" % i, (x0, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 6 if i == 2 else 2)
    fl, table = fe.analyze_typography(d, Audit(), gray=img)
    assert fl == [] and any("weight_z" in t for t in table)
    d2 = form_doc(shift={3: 130})                                         # value pushed right, but its label runs up to the column
    d2.words[6].bbox = (60, d2.words[6].bbox[1], 480, d2.words[6].bbox[3])
    assert not any("misaligned" in x.title for x in fe.analyze_typography(d2, Audit())[0])
