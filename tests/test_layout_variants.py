"""Layout variants seen in generated PDFs: diagonal watermark, label on the left with the amount far right, a bracketed note inside a
label ('Deductions (PF+Tax)'), masked account numbers, a pay month stated in the title, statement dates printed without a year.
All documents are fictional and built here."""
import pymupdf
from docforensics.pdf_layer import read_page
from docforensics.pipeline import run_case
from docforensics.extract import _masked_number, _split_qualifier
from docforensics.schema import Word


def _watermark(pg):
    pt = pymupdf.Point(120, 560)
    pg.insert_text(pt, "SYNTHETIC TEST DATA", fontname="hebo", fontsize=60, color=(0.9, 0.9, 0.9), morph=(pt, pymupdf.Matrix(35)))


def _slip(path, net):
    doc = pymupdf.open()
    pg = doc.new_page(width=595, height=842)
    _watermark(pg)
    pg.insert_text((50, 60), "ACME TEST SOLUTIONS PVT LTD (Fictional)", fontname="hebo", fontsize=16)
    pg.insert_text((50, 78), "Salary Slip for October 2026", fontname="helv", fontsize=10)
    for i, (k, v) in enumerate([("Employee Name", "Rahul Verma Test"), ("Employee ID", "TST-10234"), ("PAN", "ABCDE1234F (dummy)"),
                                ("Designation", "Analyst"), ("Bank A/C (last 4)", "XXXX 4821")]):
        pg.insert_text((50, 110 + 18 * i), k, fontname="hebo", fontsize=10)
        pg.insert_text((200, 110 + 18 * i), v, fontname="helv", fontsize=10)
    pg.insert_text((50, 210), "Earnings", fontname="hebo", fontsize=11)
    pg.insert_text((300, 210), "Amount (INR)", fontname="hebo", fontsize=11)
    for i, (k, v, b) in enumerate([("Basic", "40,000", 0), ("HRA", "16,000", 0), ("Special Allowance", "14,000", 0), ("Gross", "70,000", 1),
                                   ("Deductions (PF+Tax)", "10,000", 0)]):
        y = 230 + 16 * i
        pg.insert_text((50, y), k, fontname="hebo" if b else "helv", fontsize=10)
        pg.insert_text((369, y), v, fontname="hebo" if b else "helv", fontsize=10)
    pg.insert_text((50, 320), "Net Pay", fontname="hebo", fontsize=12)
    pg.insert_text((368, 320), net, fontname="hebo", fontsize=12)
    doc.save(path)
    doc.close()


def _bank(path, credit="52,000", closing="94,310"):
    doc = pymupdf.open()
    pg = doc.new_page(width=595, height=842)
    _watermark(pg)
    pg.insert_text((50, 60), "EXAMPLE TEST BANK (Fictional)", fontname="hebo", fontsize=16)
    pg.insert_text((50, 78), "Account Statement: 01-Oct-2026 to 31-Oct-2026", fontname="helv", fontsize=10)
    pg.insert_text((50, 106), "Account Holder: Rahul Verma Test", fontname="helv", fontsize=10)
    pg.insert_text((50, 122), "A/C No: XXXX XXXX 4821", fontname="helv", fontsize=10)
    for x, t in [(50, "Date"), (130, "Narration"), (400, "Credit"), (480, "Balance")]:
        pg.insert_text((x, 160), t, fontname="hebo", fontsize=10)
    rows = [("01-Oct", "Opening balance", "", "42,310"), ("31-Oct", "SAL ACME TEST SOLUTIONS", credit, closing)]
    for i, (d, n, c, b) in enumerate(rows):
        y = 178 + 16 * i
        pg.insert_text((50, y), d, fontname="helv", fontsize=10)
        pg.insert_text((130, y), n, fontname="helv", fontsize=10)
        if c:
            pg.insert_text((419, y), c, fontname="helv", fontsize=10)
        pg.insert_text((509, y), b, fontname="helv", fontsize=10)
    doc.save(path)
    doc.close()


def _case(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    _slip(str(d / "slip_a.pdf"), "60,000")
    _slip(str(d / "slip_b.pdf"), "95,000")
    _bank(str(d / "bank.pdf"))
    return run_case([str(d / f) for f in ("bank.pdf", "slip_a.pdf", "slip_b.pdf")], str(tmp_path / "out"), case_id="layout")


def test_diagonal_watermark_is_not_read_as_text(tmp_path):
    _slip(str(tmp_path / "s.pdf"), "60,000")
    r = read_page(str(tmp_path / "s.pdf"), 0)
    assert r["rotated_text"] and "SYNTHETIC" in r["rotated_text"][0]
    assert not any(w.text in ("SYNTHETIC", "DATA") for w in r["words"])


def test_label_variants_and_masked_account_are_extracted(tmp_path):
    res = _case(tmp_path)
    docs = {d.doc_id: d for d in res["docs"]}
    s = docs["slip_a"]
    assert (s.get("gross").value, s.get("deductions").value, s.get("net_pay").value) == (70000.0, 10000.0, 60000.0)
    assert s.get("account_no").value.endswith("4821") and s.get("account_no").value.startswith("X")
    assert s.get("pay_month").raw == "October 2026"
    b = docs["bank"]
    assert b.get("account_no").value.endswith("4821") and "01-Oct-2026" in b.get("period").raw
    assert [c["amount"] for c in b.facts["payroll_credits"]] == [52000.0]
    assert b.facts["payroll_credits"][0]["date"] == "2026-10-31"          # '31-Oct' printed without a year: taken from the statement period


def test_findings_for_changed_slip_and_bank_conflict(tmp_path):
    res = _case(tmp_path)
    titles = [f.title for f in res["findings"]]
    assert any("arithmetic" in t.lower() and "slip_b" in t for t in titles)                    # 70,000 - 10,000 != 95,000
    assert not any("arithmetic" in t.lower() and "slip_a" in t for t in titles)                # the other slip balances
    assert any(t.startswith("Payslips for the same employee and month show different net pay") for t in titles)
    assert sum(t.startswith("Salary slip net pay does not match bank payroll credit") for t in titles) == 2
    assert not any(w in t.lower() for t in titles for w in ("fraud", "forged", "forgery"))         # titles never assert fraud


def test_masked_numbers_and_qualifier_helpers():
    assert _masked_number("XXXX XXXX 4821") == "XXXXXXXX4821"
    assert _masked_number("****4821") == "XXXX4821" and _masked_number("12345678") is None
    w = lambda *t: [Word(x, (0, 0, 1, 1), 1.0) for x in t]
    q, rest = _split_qualifier(w("(PF+Tax)", "10,000"))
    assert [x.text for x in q] == ["(PF+Tax)"] and [x.text for x in rest] == ["10,000"]
    q, rest = _split_qualifier(w("(10,000)"))                                                   # a bracketed amount is a negative number, not a note
    assert q == []
