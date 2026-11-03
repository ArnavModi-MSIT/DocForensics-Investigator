"""Sections and tables: fast layout tests on fabricated word boxes, and OCR tests on generated fictional documents."""
import os, shutil, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from docforensics import extract
from docforensics.schema import Word, Field, Document
from docforensics.audit import AuditTrail
from docforensics.structure import detect_structure, table_findings, col_kind

H = 14


def put(words, text, x0, y, h=H, cw=9):
    x = x0
    for tok in text.split():
        words.append(Word(tok, (x, y, x + cw * len(tok), y + h), 0.95))
        x += cw * len(tok) + cw


def put_r(words, text, x1, y, h=H, cw=9):
    put(words, text, x1 - cw * len(text) - (text.count(" ") * cw), y, h, cw)


def doc_with(words, w=1654, h=2339, doc_type="bank_statement"):
    d = Document(doc_id="t_p1", path="t.pdf", sha256="0", image_path="x")
    d.words, d.width, d.height, d.doc_type = words, w, h, doc_type
    return d


def statement_words(last_balance=162953.65, with_total=True):
    ws = []
    put(ws, "Value", 241, 605, 17); put(ws, "Date", 141, 617, 17); put(ws, "Description", 344, 617, 17); put(ws, "Cheque", 821, 617, 17)
    put_r(ws, "Deposit", 1099, 617, 17); put_r(ws, "Withdrawal", 1338, 617, 17); put_r(ws, "Balance", 1524, 617, 17); put(ws, "Date", 249, 632, 17)
    put(ws, "16 Jun 19", 133, 665); put(ws, "16 Jun 19", 249, 665); put(ws, "BALANCE FORWARD", 338, 665); put_r(ws, "112,453.65", 1563, 665)
    put(ws, "17 Jun 19", 133, 693); put(ws, "16 Jun 19", 249, 693); put(ws, "ATM WITHDRAWAL SELF-SWITCH", 338, 693)
    put_r(ws, "1,500.00", 1369, 693); put_r(ws, "110,953.65", 1563, 693)
    put(ws, "AT NFS 04:54:54/916704002072", 338, 718)                                    # narration continues
    put(ws, "SALARY CREDIT ACME", 338, 750); put_r(ws, "52,000.00", 1170, 750); put_r(ws, "%s" % f"{last_balance:,.2f}", 1563, 750)   # no date: inherited
    if with_total:
        put(ws, "TOTAL", 725, 782); put_r(ws, "52,000.00", 1170, 782); put_r(ws, "1,500.00", 1369, 782); put_r(ws, "162,953.65", 1563, 782)
    put(ws, "Bank deposits are covered under the insurance scheme offered by DICGC", 139, 1000)   # footer ends the table
    return ws


def test_column_vocabulary():
    assert [col_kind(t) for t in ("Withdrawal Amt.", "Closing Balance", "Narration", "Chq./Ref.No.", "Value Date", "Earnings")] == \
        ["debit", "balance", "desc", "ref", "value_date", "earnings"]
    assert col_kind("Employee") is None and col_kind("Rahul") is None


def test_transaction_table_from_stacked_header_right_aligned_amounts():
    d = detect_structure(doc_with(statement_words()))
    assert len(d.tables) == 1
    t = d.tables[0]
    assert t["kind"] == "transactions" and [r["role"] for r in t["rows"]] == ["opening", "txn", "txn", "total"]
    r1, r2 = t["rows"][1], t["rows"][2]
    assert (r1["debit"], r1["balance"]) == (1500.0, 110953.65) and r1["date"] == "2019-06-17"
    assert "AT NFS" in r1["desc"]                                                      # continuation line joined to its row
    assert r2["credit"] == 52000.0 and r2["inherited_date"] and r2["date"] == "2019-06-17"
    assert t["checks"]["breaks"] == [] and t["checks"]["checked"] == 2
    assert not any(r["desc"].startswith("Bank deposits") for r in t["rows"])           # footer is not part of the table


def test_running_balance_break_is_reported_with_the_exact_arithmetic():
    d = detect_structure(doc_with(statement_words(last_balance=163953.65)))
    br = d.tables[0]["checks"]["breaks"]
    assert len(br) == 1 and br[0]["row"] == 3 and br[0]["difference"] == 1000.0 and br[0]["expected"] == 162953.65
    f = table_findings([d])
    assert len(f) == 1 and f[0].severity == "high" and "162,953.65" in f[0].explanation and "163,953.65" in f[0].explanation


def test_totals_row_must_equal_the_sum_of_all_pages():
    p1 = detect_structure(doc_with(statement_words()))
    assert table_findings([p1]) == []                                                  # 52,000 / 1,500 add up
    p1.tables[0]["rows"][3]["debit"] = 2500.0                                          # printed total no longer matches the rows
    f = table_findings([p1])
    assert len(f) == 1 and "withdrawals do not add up" in f[0].title


def test_balance_carried_across_pages():
    a = detect_structure(doc_with(statement_words()))
    b = detect_structure(doc_with(statement_words()))
    b.doc_id = "t_p2"
    b.tables[0]["rows"][0]["balance"] = 100.0                                          # page 2 starts from a different balance
    f = table_findings([a, b])
    assert any("carried forward" in x.title for x in f)


def components_words(hra=20000):
    ws = []
    put(ws, "Earnings", 70, 100, 17); put(ws, "Amount", 470, 100, 17); put(ws, "Deductions", 640, 100, 17); put(ws, "Amount", 1060, 100, 17)
    rows = [("Basic Salary", 50000, "Provident Fund", 6000), ("House Rent Allowance", hra, "Professional Tax", 200), ("Special Allowance", 25000, "Income Tax", 4800)]
    for i, (a, av, b, bv) in enumerate(rows):
        y = 150 + i * 40
        put(ws, a, 70, y, 17); put_r(ws, f"{av:,.2f}", 570, y, 17); put(ws, b, 640, y, 17); put_r(ws, f"{bv:,.2f}", 1160, y, 17)
    put(ws, "Total Earnings", 70, 270, 17); put_r(ws, "95,000.00", 570, 270, 17)
    put(ws, "Total Deductions", 640, 270, 17); put_r(ws, "11,000.00", 1160, 270, 17)
    put(ws, "Net Pay:", 70, 340, 17); put(ws, "84,000.00", 340, 340, 17)
    return ws


def test_earnings_deductions_table_and_net_pay_is_not_a_component():
    d = detect_structure(doc_with(components_words(), 1240, 1754, "salary_slip"))
    t = d.tables[0]
    assert t["kind"] == "components" and t["checks"]["mismatches"] == []
    assert not any("Net" in r["label"] for r in t["rows"])
    assert sorted(r["label"] for r in t["rows"] if r["side"] == "deductions" and r["role"] == "item") == ["Income Tax", "Professional Tax", "Provident Fund"]


def test_altered_component_breaks_the_stated_total():
    d = detect_structure(doc_with(components_words(hra=30000), 1240, 1754, "salary_slip"))
    f = table_findings([d])
    assert len(f) == 1 and "Earnings components do not add up" in f[0].title and "105,000.00" in f[0].explanation


def test_sections_from_headings_with_fields_attached_and_label_lines_not_headings():
    ws = []
    put(ws, "SALARY SLIP", 70, 60, 40, 22)
    put(ws, "EMPLOYEE DETAILS", 70, 150, 19, 12)
    put(ws, "Employee Name: Rahul Sharma", 70, 205, 17); put(ws, "Employer: Nimbus Pay", 70, 245, 17)
    put(ws, "EARNINGS AND DEDUCTIONS", 70, 380, 19, 12)
    put(ws, "Basic Salary 50,000.00", 70, 440, 17)
    d = doc_with(ws, 1240, 1754, "salary_slip")
    d.fields = [Field("name", "Employee Name", "Rahul Sharma", "Rahul Sharma", (340, 205, 520, 222), 0.95)]
    d = detect_structure(d)
    titles = [(s["title"], s["kind"]) for s in d.sections]
    assert titles == [("SALARY SLIP", "title"), ("EMPLOYEE DETAILS", "personal_details"), ("EARNINGS AND DEDUCTIONS", "earnings_deductions")]
    assert d.sections[1]["fields"] == ["name"]
    assert not any("Employee Name" in s["title"] for s in d.sections)


def test_twelve_digit_account_number_is_not_taken_for_an_aadhaar():
    audit = AuditTrail(os.path.join(os.path.dirname(__file__), "..", "output", "_t_struct_audit.jsonl"))
    ws = []
    put(ws, "Account Number:", 70, 200); put(ws, "001234567890", 340, 200)
    put(ws, "Account Holder: Rahul Sharma", 70, 240)
    d = doc_with(ws, 1240, 1754)
    extract.extract_fields(d, audit)
    assert d.get("account_no") and not d.get("aadhaar_no")


# ---- duplicated information ------------------------------------------------------------------------
def _audit():
    return AuditTrail(os.path.join(os.path.dirname(__file__), "..", "output", "_t_struct_audit.jsonl"))


def _form(lines, doc_type="application_form"):
    ws = []
    for i, text in enumerate(lines):
        put(ws, text, 70, 100 + i * 40)
    d = doc_with(ws, 1240, 1754, doc_type)
    extract.extract_fields(d, _audit())
    return d


def _titles(docs):
    from docforensics.correlate import check_intra
    return [f.title for f in check_intra(docs)]


def test_conflicting_repeat_of_a_single_valued_field_is_reported_and_first_value_kept():
    d = _form(["Date of Birth: 14-03-1998", "Applicant Name: Rahul Sharma", "Date of Birth: 15-03-1998"])
    assert d.get("dob").value == "1998-03-14"
    assert [x["key"] for x in d.facts["duplicates"]] == ["dob"]
    assert any("Two different values for 'dob'" in t for t in _titles([d]))


def test_same_value_repeated_or_recased_is_not_a_duplicate():
    d = _form(["Applicant Name: Rahul Sharma", "Date of Birth: 14-03-1998", "Date of Birth: 14-03-1998", "Applicant Name: RAHUL SHARMA"])
    assert not d.facts.get("duplicates")


# ---- misplaced information --------------------------------------------------------------------------
def test_date_under_a_name_label_is_reported_as_misplaced_not_silently_dropped():
    d = _form(["Applicant Name: 14-03-1998", "Employer: Nimbus Pay"])
    assert not d.get("name") and d.facts["rejected"][0]["kind"] == "date"
    assert any("Value under 'name' looks like a date" in t for t in _titles([d]))


def test_amount_under_a_date_label_is_misplaced_but_gibberish_is_just_unreadable():
    d = _form(["Date of Birth: Rs. 85,000"])
    assert any("looks like an amount" in t for t in _titles([d]))
    d2 = _form(["Date of Birth: ?!?"])
    assert any("could not be read" in t for t in _titles([d2])) and not any("looks like" in t for t in _titles([d2]))


def test_identity_field_inside_a_financial_section_is_flagged_but_normal_placement_is_not():
    d = doc_with([], 1240, 1754, "salary_slip")
    d.fields = [Field("name", "Employee Name", "Rahul Sharma", "Rahul Sharma", (340, 205, 520, 222), 0.95),
                Field("gross", "Gross", "95,000.00", 95000.0, (340, 500, 520, 522), 0.95)]
    mk = lambda kind, keys: {"sid": "S1", "title": kind.upper(), "kind": kind, "bbox": [0, 0, 1, 1], "n_lines": 1, "fields": keys, "tables": [], "has_heading": True}
    d.sections = [mk("earnings", ["name"])]
    assert any("'name' found inside the 'EARNINGS' section" in f.title for f in table_findings([d]))
    d.sections = [mk("personal_details", ["name"]), mk("earnings", ["gross"])]
    assert not any("found inside" in f.title for f in table_findings([d]))
    d.sections = [mk("address", ["gross"])]
    assert any("'gross' found inside" in f.title for f in table_findings([d]))


def test_fields_inside_a_table_do_not_count_as_belonging_to_the_surrounding_section():
    d = detect_structure(doc_with(statement_words()))
    d.fields = [Field("balance_forward", "BALANCE FORWARD", "112,453.65", 112453.65, (1400, 660, 1563, 680), 0.95)]
    d = detect_structure(d)
    assert all("balance_forward" not in s["fields"] for s in d.sections)


# ---- duplicated rows, references, pages ----------------------------------------------------------------
def test_identical_transaction_row_with_identical_balance_is_flagged():
    d = detect_structure(doc_with(statement_words()))
    rows = d.tables[0]["rows"]
    clone = dict(rows[1]); clone["n"] = 9
    rows.insert(2, clone)
    f = [x for x in table_findings([d]) if "Identical transaction row" in x.title]
    assert len(f) == 1 and f[0].severity == "medium"


def test_same_payment_reference_same_direction_is_flagged_but_a_reversal_is_not():
    d = detect_structure(doc_with(statement_words()))
    rows = d.tables[0]["rows"]
    rows[1]["desc"] = "UPI/917412529088/ SWIGGY"; rows[2]["desc"] = "UPI/917412529088/ SWIGGY"
    rows[1]["debit"], rows[2]["debit"], rows[2]["credit"] = 500.0, 500.0, None
    assert any("Same transaction reference" in x.title for x in table_findings([d]))
    rows[2]["debit"], rows[2]["credit"] = None, 500.0                                    # opposite direction = reversal
    assert not any("Same transaction reference" in x.title for x in table_findings([d]))


def test_near_identical_pages_are_flagged_but_similar_layouts_are_not(tmp_path):
    from PIL import Image, ImageDraw
    def page(path, shift):
        im = Image.new("RGB", (400, 300), (250, 250, 250)); dr = ImageDraw.Draw(im)
        for i in range(8):
            dr.rectangle((20 + shift, 20 + i * 30, 380, 40 + i * 30), fill=(30, 30, 30))
        im.save(path); return str(path)
    a = doc_with([], 400, 300, "salary_slip"); a.doc_id, a.image_path = "a_p1", page(tmp_path / "a.png", 0)
    b = doc_with([], 400, 300, "salary_slip"); b.doc_id, b.image_path = "b_p1", page(tmp_path / "b.png", 0)
    c = doc_with([], 400, 300, "salary_slip"); c.doc_id, c.image_path = "c_p1", page(tmp_path / "c.png", 200)
    assert any("near-identical" in f.title for f in table_findings([a, b]))
    assert not any("near-identical" in f.title for f in table_findings([a, c]))
    d = doc_with([], 400, 300, "salary_slip"); d.doc_id = "d_p1"                      # same layout, different printed content
    im = Image.new("RGB", (400, 300), (250, 250, 250)); dr = ImageDraw.Draw(im)
    for i in range(8):
        dr.rectangle((20, 20 + i * 30, 60 + i * 40, 40 + i * 30), fill=(30, 30, 30))
    im.save(tmp_path / "d.png"); d.image_path = str(tmp_path / "d.png")
    assert not any("near-identical" in f.title for f in table_findings([a, d]))


# ---- layout and hierarchy ------------------------------------------------------------------------------
def test_layout_labels_and_hierarchy_counts():
    st = detect_structure(doc_with(statement_words()))
    lay = st.facts["layout"]
    assert lay["label"] == "transaction table" and lay["hierarchy"]["tables"] == 1 and lay["hierarchy"]["table_rows"] == 4
    st.fields = [Field("name", "Name", "Rahul", "Rahul", (1, 1, 2, 2), 0.9, "fuzzy")]
    assert "header block" in detect_structure(st).facts["layout"]["label"]
    form = detect_structure(_form(["Applicant Name: Rahul Sharma", "Date of Birth: 14-03-1998", "Employer: Nimbus Pay"]))
    assert form.facts["layout"]["label"].startswith("label-value form")
    card = doc_with([], 1000, 640, "id_card"); card.fields = [Field("epic_no", "EPIC pattern", "ABC1234567", "ABC1234567", (1, 1, 2, 2), 0.9, "pattern")]
    assert detect_structure(card).facts["layout"]["label"].startswith("label-free ID card")
    assert detect_structure(doc_with([], 1000, 640, "other")).facts["layout"]["label"] == "unstructured text"


# ---- OCR on generated fictional documents ---------------------------------------------------------
def _read(path, tmp):
    from docforensics.ingest import ingest
    from docforensics.ocr import run_ocr
    audit = AuditTrail(os.path.join(tmp, "a.jsonl"))
    d = ingest([path], tmp, audit)[0]
    run_ocr(d, audit); extract.extract_fields(d, audit); extract.classify(d, audit); detect_structure(d, audit)
    return d


@pytest.fixture(scope="module")
def docs_dir(tmp_path_factory):
    from table_docs import bank_statement, salary_slip
    p = tmp_path_factory.mktemp("tbl")
    out = {"bank_ok": bank_statement(str(p / "bank_ok.jpg")),
           "bank_bal": bank_statement(str(p / "bank_bal.jpg"), tamper=("balance", 5, 66000.00)),
           "bank_amt": bank_statement(str(p / "bank_amt.jpg"), tamper=("amount", 4, 12000.00)),
           "slip_ok": salary_slip(str(p / "slip_ok.jpg")), "slip_bad": salary_slip(str(p / "slip_bad.jpg"), hra=30000)}
    return p, out


def test_ocr_genuine_statement_has_table_sections_and_no_findings(docs_dir):
    p, f = docs_dir
    d = _read(f["bank_ok"], str(p))
    assert d.doc_type == "bank_statement" and len(d.tables) == 1 and len(d.tables[0]["rows"]) >= 10
    assert [s["title"] for s in d.sections][-2:] == ["ACCOUNT DETAILS", "TRANSACTIONS"] and d.sections[0]["kind"] == "title"
    assert d.tables[0]["checks"]["breaks"] == [] and table_findings([d]) == [] and not d.get("aadhaar_no")


@pytest.mark.parametrize("name", ["bank_bal", "bank_amt"])
def test_ocr_tampered_statement_breaks_the_running_balance(docs_dir, name):
    p, f = docs_dir
    d = _read(f[name], str(p))
    findings = table_findings([d])
    assert any("Running balance does not reconcile" in x.title for x in findings), [x.title for x in findings]


def test_ocr_salary_slip_components(docs_dir):
    p, f = docs_dir
    ok, bad = _read(f["slip_ok"], str(p)), _read(f["slip_bad"], str(p))
    assert ok.tables and ok.tables[0]["kind"] == "components" and table_findings([ok]) == []
    assert any("Earnings components" in x.title for x in table_findings([bad]))
