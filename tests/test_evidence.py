"""Evidence mapping (5.7): crops, side-by-side montage for cross-document findings, status label, why / check text."""
import os
from PIL import Image
from docforensics.schema import Document, Finding
from docforensics.evidence import build_evidence, guidance


def _doc(tmp_path, name):
    p = tmp_path / f"{name}.png"
    Image.new("RGB", (800, 600), (240, 240, 240)).save(p)
    return Document(doc_id=name, path=name, sha256="0", image_path=str(p), width=800, height=600)


def test_cross_document_finding_gets_one_montage_with_both_regions(tmp_path):
    a, b = _doc(tmp_path, "a"), _doc(tmp_path, "b")
    f = Finding("X-001", "cross", "t", "high", 0.9, "inconsistency", "e",
                [{"doc_id": "a", "field": "dob", "value": "19 Apr", "bbox": [100, 100, 220, 130]},
                 {"doc_id": "b", "field": "dob", "value": "18 Apr", "bbox": [300, 300, 420, 330]}], method="cross-document dob comparison")
    ev = build_evidence([a, b], [f], str(tmp_path))["X-001"]
    assert ev["regions"] == 2 and os.path.exists(tmp_path / ev["crop"])
    assert ev["label"].startswith("Confirmed data conflict") and "intent" in ev["label"] and ev["why"] and ev["check"]
    assert Image.open(tmp_path / ev["crop"]).width > 300


def test_whole_page_finding_has_text_but_no_crop(tmp_path):
    a = _doc(tmp_path, "a")
    f = Finding("C-001", "missing", "t", "low", 0.7, "observed", "e", [{"doc_id": "a", "field": None, "value": None, "bbox": None}], method="x")
    ev = build_evidence([a], [f], str(tmp_path))["C-001"]
    assert ev["crop"] is None and ev["regions"] == 0 and ev["label"].startswith("Observation") and "not evidence" in ev["why"]


def test_potential_alteration_label_never_says_fraud():
    for st in ("inconsistency", "potential_manipulation", "observed"):
        label, why, check = guidance(Finding("V-1", "visual", "t", "low", 0.5, st, "e", [], method="PDF structure: fonts"))
        assert "fraud" not in (label + why + check).lower()
    assert guidance(Finding("V-1", "visual", "t", "low", 0.5, "potential_manipulation", "e", [], method="m"))[0].startswith("Potential alteration")
