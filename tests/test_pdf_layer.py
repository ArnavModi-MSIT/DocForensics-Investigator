"""Born-digital PDFs: text layer, stacked (label above value) extraction, PDF structure forensics, corroboration rules.
All documents are generated here with fictional data."""
import pymupdf
import pytest
from docforensics.audit import AuditTrail
from docforensics.pipeline import run_case
from docforensics.extract import looks_like_label
from docforensics.correlate import corroborate
from docforensics.schema import Finding


def _page(path, gross="INR 97,000", ded="INR 7,300", net="INR 89,700", patch=False):
    """Fictional payslip with labels ABOVE values, in the style of key-value grids."""
    doc = pymupdf.open()
    pg = doc.new_page(width=595, height=842)
    pg.draw_rect(pymupdf.Rect(0, 0, 595, 100), fill=(0.08, 0.15, 0.24), color=None)
    pg.insert_text((43, 45), "NORTHWIND DEMO LABS", fontname="hebo", fontsize=14, color=(1, 1, 1))
    pg.insert_text((43, 72), "Salary Statement | October 2026", fontname="helv", fontsize=12, color=(1, 1, 1))

    def pair(x, y, label, value):
        pg.insert_text((x, y), label, fontname="hebo", fontsize=7.4, color=(0.4, 0.45, 0.5))
        pg.insert_text((x, y + 20), value, fontname="helv", fontsize=10, color=(0.05, 0.1, 0.15))
    pair(54, 160, "EMPLOYEE NAME", "Asha Verma")
    pair(326, 160, "EMPLOYEE ID", "NW-3141")
    pair(54, 220, "PAYROLL PERIOD", "October 2026")
    pair(326, 220, "PAYMENT DATE", "30 Oct 2026")
    for i, (a, b) in enumerate([("Base pay", "78,000"), ("Allowance", "12,000"), ("Other", "7,000")]):
        y = 330 + 28 * i
        pg.insert_text((58, y), a, fontname="helv", fontsize=9.7)
        pg.insert_text((243, y), b, fontname="helv", fontsize=9.7)
    for i, (a, b) in enumerate([("Retirement", "4,200"), ("Income tax", "3,100")]):
        y = 330 + 28 * i
        pg.insert_text((328, y), a, fontname="helv", fontsize=9.7)
        pg.insert_text((508, y), b, fontname="helv", fontsize=9.7)
    pg.insert_text((58, 340 - 40), "EARNINGS", fontname="hebo", fontsize=8.6)
    pg.insert_text((260, 340 - 40), "INR", fontname="hebo", fontsize=8.6)
    pg.insert_text((328, 340 - 40), "DEDUCTIONS", fontname="hebo", fontsize=8.6)
    pg.insert_text((519, 340 - 40), "INR", fontname="hebo", fontsize=8.6)
    pair(65, 440, "GROSS PAY", gross)
    pair(265, 440, "TOTAL DEDUCTIONS", ded)
    pg.draw_rect(pymupdf.Rect(55, 500, 543, 572), fill=(0.88, 0.93, 0.95), color=None)
    pg.insert_text((69, 520), "NET PAY / TAKE-HOME", fontname="hebo", fontsize=10)
    if patch:
        pg.draw_rect(pymupdf.Rect(327, 508, 499, 539), fill=(0.96, 0.95, 0.91), color=(0.83, 0.81, 0.75), width=0.6)
        pg.insert_text((344, 530), net, fontname="cour", fontsize=15.6)
    else:
        pg.insert_text((337, 530), net, fontname="hebo", fontsize=15.6)
    doc.save(path)
    doc.close()


def _run(tmp_path, name, **kw):
    p = tmp_path / f"{name}.pdf"
    _page(str(p), **kw)
    return run_case([str(p)], str(tmp_path / f"out_{name}"), case_id=name)


def test_looks_like_label():
    assert looks_like_label("DATE OF BIRTH")
    assert looks_like_label("MONTHLY GROSS SALARY")
    assert not looks_like_label("24 months")             # digits: a value
    assert not looks_like_label("Kavya Mehta")


def test_text_layer_and_stacked_extraction(tmp_path):
    r = _run(tmp_path, "clean")
    d = r["docs"][0]
    assert d.born_digital and d.facts["ocr_engine"] == "pdf-text-layer" and d.facts["ocr_mean_conf"] == 1.0
    got = {f.key: f.raw for f in d.fields}
    assert got["name"] == "Asha Verma" and got["employee_id"] == "NW-3141"
    assert got["gross"] == "INR 97,000" and got["deductions"] == "INR 7,300" and got["net_pay"] == "INR 89,700"
    assert got["pay_date"] == "30 Oct 2026"
    assert all(f.match in ("stacked", "fuzzy", "fuzzy-cell", "header") for f in d.fields)


def test_clean_pdf_has_no_substantive_findings(tmp_path):
    r = _run(tmp_path, "clean2")
    bad = [f for f in r["findings"] if f.severity != "low" or f.status != "observed"]
    assert not bad, [f.title for f in bad]


def test_patch_and_font_change_are_reported_as_observation_with_arithmetic(tmp_path):
    r = _run(tmp_path, "patched", net="INR 99,700", patch=True)
    titles = {f.fid: f for f in r["findings"]}
    vis = [f for f in r["findings"] if f.kind == "visual"]
    assert vis and vis[0].status == "potential_manipulation" and "patch" in vis[0].title.lower()
    assert "not" in vis[0].explanation.lower() and "proof" not in vis[0].title.lower()
    ar = [f for f in r["findings"] if "arithmetic" in f.title]
    assert ar and ar[0].severity == "high" and vis[0].fid in ar[0].metrics.get("corroborated_by", [])
    assert not any("fraud" in f.title.lower() for f in r["findings"])


def test_whole_page_references_never_corroborate():
    ref_page = {"doc_id": "a", "field": None, "value": None, "bbox": None}
    vis = Finding("V-001", "visual", "x", "low", 0.5, "potential_manipulation", "", [dict(ref_page)])
    other = Finding("C-001", "missing", "y", "low", 0.7, "observed", "", [dict(ref_page)])
    corroborate([vis], [other])
    assert other.severity == "low" and "corroborated_by" not in other.metrics


def test_observation_only_visual_finding_does_not_raise_severity():
    box = {"doc_id": "a", "field": "net_pay", "value": "1", "bbox": [0, 0, 10, 10]}
    vis = Finding("V-001", "visual", "x", "low", 0.4, "observed", "", [dict(box)])
    other = Finding("C-001", "intra", "y", "medium", 0.7, "inconsistency", "", [dict(box)])
    corroborate([vis], [other])
    assert other.severity == "medium"
