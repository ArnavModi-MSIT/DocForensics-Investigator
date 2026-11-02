"""OCR engine layer: variant scoring, RapidOCR word splitting, engine selection, busy-background recovery.
RapidOCR-dependent tests are skipped when `rapidocr-onnxruntime` is not installed (it is optional)."""
import os, sys, shutil
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from docforensics import ocr
from docforensics.schema import Word
from docforensics.audit import AuditTrail
from docforensics.ingest import ingest
from docforensics.extract import extract_fields, classify
from busy_card import busy_card

needs_rapid = pytest.mark.skipif(not ocr.rapidocr_available(), reason="rapidocr-onnxruntime not installed")


def W(t, c=0.9):
    return Word(t, (0, 0, 10, 10), c)


def test_respace_splits_glued_labels_but_keeps_names():
    assert ocr._respace("SexMale") == ["Sex", "Male"]
    assert ocr._respace("DateofBirth:")[:2] == ["date", "of"]
    assert ocr._respace("Date6fBirth:")[-1] == "birth:"            # OCR slip tolerated
    assert ocr._respace("McDonald") == ["McDonald"]
    assert ocr._respace("MacArthur", first=False) == ["MacArthur"]


def test_score_prefers_clean_reading_over_junk_volume():
    clean = [W(t) for t in ("Name", "Rahul", "Sharma", "Date", "Birth", "14-03-1998")]
    junk = [W(t, 0.25) for t in ("ae", "yy", "4", "~", "fa", "vA")] * 60
    assert ocr._score(clean) > ocr._score(junk + clean[:2])
    assert ocr._poor(junk) and not ocr._poor(clean * 2)


def test_rapid_word_boxes_are_proportional(monkeypatch):
    class Fake:
        def __call__(self, img):
            return [([[10, 0], [110, 0], [110, 20], [10, 20]], "Name:Rahul Sharma", 0.9)]
    monkeypatch.setattr(ocr, "_get_rapid", lambda: Fake())
    ws = ocr._rapid_words(None)
    assert [w.text for w in ws] == ["Name:", "Rahul", "Sharma"]
    assert ws[0].bbox[0] == 10 and ws[-1].bbox[2] <= 110 and ws[0].bbox[2] <= ws[1].bbox[0] + 1


def _run(path, mode, tmp_path, monkeypatch):
    monkeypatch.setenv("DOCFOR_OCR", mode)
    wd = str(tmp_path / mode); os.makedirs(wd, exist_ok=True)
    a = AuditTrail(os.path.join(wd, "audit.jsonl"))
    docs = ingest([path], wd, a)
    for d in docs:
        ocr.run_ocr(d, a); extract_fields(d, a); classify(d, a)
    return docs[0]


def test_tesseract_mode_never_retries(tmp_path, monkeypatch):
    p = busy_card(str(tmp_path / "c.jpg"), 0.5)
    monkeypatch.setattr(ocr, "_get_rapid", lambda: pytest.fail("RapidOCR must not run in tesseract mode"))
    d = _run(p, "tesseract", tmp_path, monkeypatch)
    assert d.facts["ocr_engine"] == "tesseract"


def test_engine_failure_does_not_kill_pipeline(tmp_path, monkeypatch):
    class Boom:
        def __call__(self, img): raise RuntimeError("model crashed")
    p = busy_card(str(tmp_path / "c.jpg"), 0.5)
    monkeypatch.setattr(ocr, "_get_rapid", lambda: Boom())
    d = _run(p, "auto", tmp_path, monkeypatch)
    assert d.words and d.facts["ocr_engine"].startswith("tesseract")


@needs_rapid
def test_busy_background_recovered_by_rapidocr(tmp_path, monkeypatch):
    p = busy_card(str(tmp_path / "c.jpg"), 0.5)
    auto = _run(p, "auto", tmp_path, monkeypatch)
    assert auto.facts["ocr_engine"] == "rapidocr"
    got = {f.key: f.value for f in auto.fields}
    assert got.get("epic_no") == "ABC1234567"
    assert got.get("dob") == "1998-03-14"
    assert got.get("father_name", "").startswith("Suresh Sh")
    assert auto.fields and len(auto.words) < 60                   # clean reading, not hundreds of junk tokens
