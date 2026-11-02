"""Fast unit tests (no OCR). Each confirmed defect from the external reviews has a regression test here.
Run: pytest -q"""
import io, os, zipfile, json
import numpy as np
import pytest
from PIL import Image, ImageDraw
from docforensics import extract, correlate, ingest, dossier, id_patterns
from docforensics.audit import AuditTrail, verify_chain
from docforensics.schema import Word, Field, Document


def W(t, x, y, c=0.95):
    return Word(t, (x, y, x + len(t) * 14, y + 22), c)


def F(k, raw, val, label=None):
    return Field(k, label or k, raw, val, (0, 0, 10, 10), 0.9)


def D(i, t, fields):
    d = Document(doc_id=i, path=i, sha256="x", image_path="", doc_type=t)
    d.fields = fields
    return d


@pytest.fixture
def audit(tmp_path):
    return AuditTrail(str(tmp_path / "a.jsonl"))


# ---------------------------------------------------------------- parsing and validation
@pytest.mark.parametrize("s", ["14/03/1994", "14-03-1994", "14.03.1994", "14-MAR-1994", "14 March 1994", "14/Mar/94", "1994-03-14"])
def test_parse_date_formats(s):
    assert extract.parse_date(s) == "1994-03-14"


def test_parse_date_rejects_garbage():
    assert extract.parse_date("INVALID") is None


def test_parse_money_and_ocr_repair():
    assert extract.parse_money("Rs. 5,000") == 5000.0
    assert extract.parse_money("1O0.00") == 100.0          # OCR repair only when plain parsing fails
    assert extract.parse_money("NOT_A_NUMBER") is None
    assert extract.parse_money("OOO") is None               # no real digit: not a number


def test_finalize_value_cuts_trailing_junk():
    assert extract.finalize_value("deductions", "Rs. 5,000 i i") == ("Rs. 5,000", 5000.0)
    assert extract.finalize_value("dob", "14-MAR-1994 xx") [1] == "1994-03-14"
    assert extract.finalize_value("dob", "INVALID") is None
    assert extract.finalize_value("gross", "NOT_A_NUMBER") is None
    assert extract.finalize_value("id_number", "PLACEHOLDER") is None
    assert extract.finalize_value("id_number", "RPT-20260929-150736") is None      # report code, not an ID


def test_cleaner_never_takes_numbers_from_other_text():
    assert extract.finalize_value("gross", "$577.3K") is None                       # number runs into other text
    assert extract.finalize_value("gross", "about 25,981 per month") is None        # value must start the string
    assert extract.finalize_value("id_number", "boundaries computed from 20260929") is None
    assert extract.finalize_value("dob", "see 14/03/1994 for details") is None
    assert extract.finalize_value("account_no", "5 lakh customers") is None
    assert extract.finalize_value("id_number", "ABCDE1234F") == ("ABCDE1234F", "ABCDE1234F")
    assert extract.finalize_value("account_no", "5010 0234 5678 91")[1] == "50100234567891"


def test_keyword_whole_word():
    assert extract.keyword_hits("a proof of issues", ["issue"]) == []
    assert extract.keyword_hits("an applicant-facing app", ["applicant"]) == []
    assert extract.keyword_hits("date of issue", ["issue"]) == ["issue"]


def test_first_pass_rejects_malformed_values(audit):
    d = Document(doc_id="m", path="m", sha256="x", image_path="", doc_type="unknown")
    d.words = [W("Date", 60, 100), W("of", 120, 100), W("Birth:", 160, 100), W("INVALID", 430, 100),
               W("Gross", 60, 200), W("Earnings:", 130, 200), W("NOT_A_NUMBER", 430, 200),
               W("Total", 60, 400), W("Deductions:", 130, 400), W("Rs.", 430, 400), W("5,000", 470, 400), W("i", 540, 400)]
    extract.extract_fields(d, audit)
    keys = {f.key: f for f in d.fields}
    assert "dob" not in keys and "gross" not in keys                      # invalid values are not stored as fields
    assert keys["deductions"].value == 5000.0
    assert {u["key"] for u in d.facts["unparsed"]} >= {"dob", "gross"}     # ...but they are reported
    assert d.facts["ocr_cleaned"][0]["original"].startswith("Rs. 5,000")  # and the correction is recorded


# ---------------------------------------------------------------- id patterns
def test_verhoeff_roundtrip_and_single_digit_errors():
    base = "29834756102"
    n = base + str(id_patterns.verhoeff_check_digit(base))
    assert id_patterns.verhoeff_valid(n)
    for pos in range(12):
        for dgt in "0123456789":
            if dgt != n[pos]:
                assert not id_patterns.verhoeff_valid(n[:pos] + dgt + n[pos + 1:])


def test_find_aadhaar_ignores_vid_and_flags_bad_checksum():
    base = "29834756102"
    n = base + str(id_patterns.verhoeff_check_digit(base))
    ok = id_patterns.find_aadhaar([[W(n[:4], 0, 0), W(n[4:8], 80, 0), W(n[8:], 160, 0)]])
    assert len(ok) == 1 and ok[0]["checksum_ok"]
    bad = id_patterns.find_aadhaar([[W(n[:4], 0, 0), W(n[4:8], 80, 0), W("0000", 160, 0)]])
    assert bad[0]["checksum_ok"] is False
    assert id_patterns.find_aadhaar([[W("9111", 0, 0), W("2222", 80, 0), W("3333", 160, 0), W("4444", 240, 0)]]) == []


# ---------------------------------------------------------------- correlation
def test_zero_amount_does_not_crash():
    form = D("form", "application_form", [F("declared_income", "Rs. 85,000", 85000.0)])
    bank = D("bank", "bank_statement", [F("salary_credit", "Rs. 0", 0.0)])
    titles = [f.title for f in correlate.check_cross([form, bank])]
    assert any("Declared income exceeds" in t for t in titles)


def test_generic_id_is_not_validated_as_pan():
    passport = D("p", "id_card", [F("id_number", "K1234567", "K1234567", label="Passport No")])
    assert not [f for f in correlate.check_intra([passport]) if "format" in f.title.lower()]
    pan_bad = D("q", "id_card", [F("id_number", "ABCDE12345", "ABCDE12345", label="PAN")])
    assert [f for f in correlate.check_intra([pan_bad]) if "format" in f.title.lower()]


def test_unreadable_value_is_reported_not_silent():
    d = D("slip", "salary_slip", [F("name", "Rohan Mehta", "Rohan Mehta")])
    d.facts["unparsed"] = [{"key": "net_pay", "raw": "NOT_A_NUMBER"}, {"key": "net_pay", "raw": "NOT_A_NUMBER"}]
    fs = [f for f in correlate.check_intra([d]) if "could not be read" in f.title]
    assert len(fs) == 1                                                    # reported once


def test_odd_one_out_names():
    docs = [D(n, t, [F("name", v, v)]) for n, t, v in [("a", "application_form", "Rohan Mehta"), ("b", "id_card", "Rohan Mehta"),
                                                        ("c", "salary_slip", "Rohan Mehta"), ("d", "bank_statement", "Rohan Mehra")]]
    fs = correlate.check_cross(docs)
    assert len(fs) == 1 and "disagrees" in fs[0].title


# ---------------------------------------------------------------- ingestion
def _jpg(color):
    b = io.BytesIO()
    Image.new("RGB", (50, 50), color).save(b, "JPEG")
    return b.getvalue()


def test_zip_name_collision_keeps_both_files(tmp_path, audit):
    zp = tmp_path / "t.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("a/b.jpg", _jpg((255, 0, 0)))
        z.writestr("a__b.jpg", _jpg((0, 0, 255)))
    paths, _ = ingest.expand_inputs([str(zp)], str(tmp_path), audit)
    assert len(paths) == 2 and len(set(paths)) == 2
    assert len({open(p, "rb").read() for p in paths}) == 2


def test_zip_rejects_unsafe_and_unsupported(tmp_path, audit):
    zp = tmp_path / "t.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("ok.jpg", _jpg((1, 2, 3)))
        z.writestr("../evil.jpg", _jpg((1, 2, 3)))
        z.writestr("/abs.jpg", _jpg((1, 2, 3)))
        z.writestr("notes.txt", "x")
    paths, skipped = ingest.expand_inputs([str(zp)], str(tmp_path), audit)
    assert len(paths) == 1 and {s["reason"] for s in skipped} >= {"unsafe path", "unsupported type .txt"}
    assert not (tmp_path.parent / "evil.jpg").exists()


def test_exif_orientation_applied(tmp_path):
    ex = Image.Exif()
    ex[0x0112] = 6
    p = tmp_path / "e.jpg"
    Image.new("RGB", (100, 200), "white").save(p, "JPEG", exif=ex)
    assert ingest.load_pages(str(p))[0].size == (200, 100)


def test_same_stem_gets_unique_doc_ids(tmp_path, audit):
    a, b = tmp_path / "x" / "scan.jpg", tmp_path / "y" / "scan.png"
    a.parent.mkdir(); b.parent.mkdir()
    Image.new("RGB", (40, 40), "white").save(a, "JPEG")
    Image.new("RGB", (40, 40), "black").save(b, "PNG")
    docs = ingest.ingest([str(a), str(b)], str(tmp_path / "w"), audit)
    assert len({d.doc_id for d in docs}) == 2 and len({d.image_path for d in docs}) == 2


# ---------------------------------------------------------------- reporting, privacy, audit
def test_pdf_survives_markup_in_text(tmp_path):
    f = {"fid": "X-001", "kind": "cross", "title": "Name: <Rohan & <b>", "severity": "high", "confidence": 0.9,
         "status": "inconsistency", "explanation": "value was <unclosed", "refs": [{"doc_id": "a<b", "field": "name", "value": "<x", "bbox": None}],
         "metrics": {}, "method": "m<"}
    dos = {"case_id": "c<1>", "audit_digest": "0" * 64, "summary": "s <b", "summary_source": "template", "documents": [], "findings": [f],
           "capabilities": [{"capability": "c<", "status": "s&"}], "audit_trail": []}
    out = dossier.build_pdf(dos, {}, str(tmp_path / "d.pdf"))
    assert os.path.getsize(out) > 1000


def test_aadhaar_redacted_in_evidence_image():
    im = Image.new("RGB", (600, 100), "white")
    ImageDraw.Draw(im).rectangle((100, 30, 500, 60), fill=(200, 0, 0))
    d = D("c", "id_card", [Field("aadhaar_no", "x", "XXXX XXXX 1234", "123456781234", (100, 30, 500, 60), 0.9)])
    dossier.redact_ids(d, im)
    assert im.getpixel((150, 45)) == (0, 0, 0)               # first 8 digits blacked out
    assert im.getpixel((490, 45)) == (200, 0, 0)             # last 4 remain visible


def test_audit_chain_detects_tampering(tmp_path):
    a = AuditTrail(str(tmp_path / "a.jsonl"))
    for i in range(5):
        a.log("step", i=i)
    assert verify_chain(a.events)
    ev = [dict(e) for e in a.events]
    ev[2]["i"] = 99
    assert not verify_chain(ev)
    assert not verify_chain([ev[0], a.events[2]] + a.events[3:])          # removed event


# ---------------------------------------------------------------- Indian ID layouts (mock OCR lines; no real cards)
def _line(y, items):
    """items: [(x, text, conf)] -> Word list on one visual line."""
    return [Word(t, (x, y, x + len(t) * 14, y + 22), c) for x, t, c in items]


def _doc(lines, name="card"):
    d = Document(doc_id=name, path=name, sha256="x", image_path="", doc_type="unknown")
    d.words = [w for ln in lines for w in ln]
    return d


def _run(d, audit):
    extract.extract_fields(d, audit)
    extract.classify(d, audit)
    return {f.key: f for f in d.fields}


def test_voter_id_layout_with_bilingual_noise(audit):
    lines = [_line(40, [(100, "ELECTION", .95), (200, "COMMISSION", .95), (330, "OF", .95), (370, "INDIA", .95)]),
             _line(80, [(100, "Elector", .95), (180, "Photo", .95), (250, "Identity", .95), (330, "Card", .95)]),
             _line(130, [(60, "ABC1234567", .95)]),
             _line(200, [(40, "ara:", .3), (300, "Name:", .95), (370, "Rahul", .95), (440, "Sharma", .95)]),
             _line(250, [(40, "pita", .3), (300, "Father's", .95), (380, "Name:", .95), (450, "Suresh", .95), (520, "Sharma", .95)]),
             _line(300, [(300, "Date", .95), (350, "of", .95), (380, "Birth", .95), (430, "/", .95), (450, "Age:", .95), (500, "14-03-1998", .95)])]
    d = _doc(lines)
    f = _run(d, audit)
    assert f["epic_no"].value == "ABC1234567"
    assert f["name"].value == "Rahul Sharma" and f["father_name"].value == "Suresh Sharma"
    assert f["dob"].value == "1998-03-14"
    assert d.doc_type == "id_card"


def test_pan_card_stacked_layout_without_labels(audit):
    lines = [_line(40, [(100, "INCOME", .95), (180, "TAX", .95), (230, "DEPARTMENT", .95), (360, "GOVERNMENT", .95), (480, "OF", .95), (520, "INDIA", .95)]),
             _line(200, [(60, "SAMPLE", .95), (150, "NAME", .95)]),
             _line(250, [(60, "SAMPLE", .95), (150, "FATHER", .95)]),
             _line(300, [(60, "01/01/1990", .95)]),
             _line(350, [(60, "ABCDE1234F", .95)]),
             _line(420, [(500, "Sanple!", .12), (580, "Nae", .12)])]
    d = _doc(lines)
    f = _run(d, audit)
    assert f["id_number"].value == "ABCDE1234F"
    assert (f["name"].value, f["father_name"].value, f["dob"].value) == ("SAMPLE NAME", "SAMPLE FATHER", "1990-01-01")
    assert d.doc_type == "id_card"


def test_headers_are_not_taken_as_names(audit):
    lines = [_line(60, [(60, "INCOME", .95), (140, "TAX", .95)]), _line(110, [(60, "GOVERNMENT", .95), (190, "OF", .95), (230, "INDIA", .95)]),
             _line(160, [(60, "01/01/1990", .95)]), _line(210, [(60, "ABCDE1234F", .95)])]
    f = _run(_doc(lines), audit)
    assert "name" not in f and "father_name" not in f


def test_back_of_card_follows_front_by_same_id(audit):
    front = _doc([_line(130, [(60, "ABC1234567", .95)])], "front")
    extract.extract_fields(front, audit)
    front.doc_type = "id_card"
    back = _doc([_line(130, [(60, "ABC1234567", .95)]), _line(300, [(60, "Address:", .95), (160, "Flat", .95), (210, "12,", .95), (290, "201301", .95)])], "back")
    back.path = front.path                                   # pages of the same file
    extract.extract_fields(back, audit)
    back.class_info = {"keyword": {"id_card": 0.0, "application_form": 0.0, "salary_slip": 0.0, "bank_statement": 0.0}}
    extract.inherit_continuations([front, back], audit)
    assert back.doc_type == "id_card" and "same ID number" in back.class_info["decision"]


def test_different_id_kinds_are_never_compared():
    form = D("form", "application_form", [F("id_number", "ABCDE1234F", "ABCDE1234F")])
    voter = D("voter", "id_card", [F("epic_no", "ABC1234567", "ABC1234567")])
    assert not [f for f in correlate.check_cross([form, voter]) if "number" in f.title.lower()]
    voter2 = D("voter2", "id_card", [F("epic_no", "ABC1234568", "ABC1234568")])
    assert [f for f in correlate.check_cross([voter, voter2]) if "EPIC" in f.title]


def test_textured_background_is_not_judged_by_forensics(tmp_path, audit):
    from PIL import ImageFont
    from docforensics import forensics
    fp = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if not os.path.exists(fp):
        pytest.skip("DejaVu font not available")
    rng = np.random.default_rng(1)
    arr = np.clip(200 + rng.normal(0, 18, (700, 900, 3)), 0, 255).astype(np.uint8)        # decorated card-like background
    im = Image.fromarray(arr)
    dr = ImageDraw.Draw(im)
    ft = ImageFont.truetype(fp, 30)
    d = Document(doc_id="card", path="c", sha256="x", image_path=str(tmp_path / "card.png"), doc_type="id_card")
    for i, k in enumerate(["name", "father_name", "dob", "id_number", "address"]):
        y = 60 + i * 110
        dr.text((100, y), f"VALUE {i} ABCDEFG", font=ft, fill=(10, 10, 10))
        d.fields.append(Field(k, k, "x", "x", dr.textbbox((100, y), f"VALUE {i} ABCDEFG", font=ft), 0.9))
    im.save(d.image_path)
    findings, table = forensics.analyze(d, audit)
    assert findings == [] and table == []
    assert "textured background" in d.facts["forensics_reason"]


def test_confident_punctuation_is_kept_in_values(audit):
    d = _doc([_line(100, [(60, "Statement", .95), (170, "Period:", .95), (300, "01/10/2026", .95), (420, "-", .93), (440, "31/10/2026", .95)])])
    extract.extract_fields(d, audit)
    assert {f.key: f.raw for f in d.fields}["period"] == "01/10/2026 - 31/10/2026"


def test_date_with_ocr_spacing_mixed_separators_and_symbol_noise():
    from docforensics.extract import finalize_value
    assert finalize_value("dob", "14- 03.1998 ff «7") == ("14- 03.1998", "1998-03-14")
    assert finalize_value("dob", "14-03-94")[1] == "1994-03-14"
    assert finalize_value("dob", "14-03-1998 Male 1234") is None            # real numbers after the date are still refused


def test_designation_lines_are_not_person_names():
    from docforensics.extract import _value_ok
    assert not _value_ok("name", "Officer, 40 - NEW DELHI (GEN)")
    assert not _value_ok("father_name", "Electoral Registration Officer")
    assert _value_ok("name", "Rahul Sharma")


def test_unparsed_value_reported_once_per_field():
    from docforensics.correlate import check_intra
    from docforensics.schema import Document
    d = Document(doc_id="x", path="x", sha256="0", image_path="x")
    d.doc_type = "id_card"
    d.facts["unparsed"] = [{"key": "dob", "raw": "14- 03.1998 ff"}, {"key": "dob", "raw": "14- 03.1998"}]
    assert sum("'dob' could not be read" in f.title for f in check_intra([d])) == 1


def test_summary_does_not_claim_visual_clean_when_forensics_never_ran():
    from docforensics.llm import template_summary
    d = Document(doc_id="x", path="x", sha256="0", image_path="x")
    d.doc_type = "id_card"; d.facts["forensics"] = "not run: textured background"
    s = template_summary("c1", [d], [])
    assert "no visual-tampering check was possible" in s.lower() and "visual anomalies were detected" not in s


# ---- per-field cross-document comparators -------------------------------------------------------
@pytest.mark.parametrize("key,a,b,verdict", [
    ("name", "R. K. Sharma", "Rahul Kumar Sharma", "compatible"),
    ("name", "Sharma, Rahul", "Rahul Sharma", "same"),
    ("name", "Mr Rahul Sharma", "RAHUL SHARMA", "same"),
    ("name", "Rahul Sharma", "Rahul Kumar Sharma", "compatible"),
    ("name", "Rahul", "Rahul Kumar Sharma", "unverifiable"),
    ("name", "Rahul Sharma", "Rohan Verma", "different"),
    ("name", "Rahul Sharma", "Rahul Sharna", "near"),
    ("address", "Flat 12, Sector 21, Noida, Uttar Pradesh 201301", "12 Sector-21 Noida 201301", "compatible"),
    ("address", "Flat 12 Sector 21 Noida 201301", "Flat 12 Sector 21 Noida 201302", "different"),
    ("address", "House 5 MG Road Pune 411001", "Plot 9 Park Street Kolkata 411001", "different"),
    ("address", "Noida", "Delhi", "unverifiable"),
    ("employer", "Acme Pvt Ltd", "ACME PRIVATE LIMITED", "same"),
    ("employer", "Acme Solutions Pvt. Ltd.", "Acme Solutions", "same"),
    ("employer", "Acme Solutions", "Zenith Corp", "different"),
    ("employer", "Tata Consultancy Services", "Tata Consultancy Servces", "near"),
    ("account_no", "XXXXXX1234", "001234561234", "compatible"),
    ("account_no", "001234561234", "001234561235", "near"),
    ("id_number", "ABCDE1234F", "ABCDE1234F", "same"),
])
def test_field_comparators(key, a, b, verdict):
    from docforensics.compare import compare
    assert compare(key, a, b).verdict == verdict


def _fdoc(i, typ, **fields):
    d = Document(doc_id=i, path=i, sha256=i, image_path=i)
    d.doc_type = typ
    for k, v in fields.items():
        d.fields.append(Field(key=k, label=k, raw=str(v), value=v, bbox=(0, 0, 1, 1), conf=0.95, match="fuzzy", label_sim=100))
    return d


def test_cross_doc_consistent_variants_raise_no_finding():
    a = _fdoc("form", "application_form", name="R. K. Sharma", address="Flat 12, Sector 21, Noida 201301", employer="Acme Pvt Ltd")
    b = _fdoc("slip", "salary_slip", name="Rahul Kumar Sharma", employer="ACME PRIVATE LIMITED")
    c = _fdoc("id", "id_card", name="Rahul Kumar Sharma", address="12 Sector-21 Noida, Uttar Pradesh 201301")
    assert correlate.check_cross([a, b, c]) == []


def test_cross_doc_real_mismatch_and_low_confidence_downgrade():
    a = _fdoc("form", "application_form", name="Rahul Sharma")
    b = _fdoc("slip", "salary_slip", name="Rohan Verma")
    f = correlate.check_cross([a, b])
    assert len(f) == 1 and f[0].severity == "high"
    b.fields[0].conf = 0.4
    assert correlate.check_cross([a, b])[0].severity == "medium"


def test_unverifiable_pair_is_reported_as_coverage_not_mismatch():
    a = _fdoc("form", "application_form", name="Rahul")
    b = _fdoc("id", "id_card", name="Rahul Kumar Sharma")
    f = correlate.check_cross([a, b])
    assert [x.fid for x in f] == ["N-004"]


def test_declared_gross_is_not_flagged_when_slip_gross_supports_it():
    form = _fdoc("form", "application_form", declared_income=100000.0)
    slip = _fdoc("slip", "salary_slip", net_pay=80000.0, gross=100000.0)
    bank = _fdoc("bank", "bank_statement", salary_credit=80000.0)
    assert correlate.check_cross([form, slip, bank]) == []
    slip2 = _fdoc("slip", "salary_slip", net_pay=80000.0, gross=90000.0)
    f = correlate.check_cross([form, slip2, bank])
    assert len(f) == 1 and f[0].severity == "high"


def test_net_only_evidence_gives_medium_gross_net_caveat():
    form = _fdoc("form", "application_form", declared_income=100000.0)
    bank = _fdoc("bank", "bank_statement", salary_credit=80000.0)
    f = correlate.check_cross([form, bank])
    assert len(f) == 1 and f[0].severity == "medium" and "gross" in f[0].explanation.lower()


def test_multiple_payslips_each_matched_to_some_credit():
    s1 = _fdoc("s1", "salary_slip", net_pay=80000.0); s2 = _fdoc("s2", "salary_slip", net_pay=82000.0)
    b1 = _fdoc("b1", "bank_statement", salary_credit=80000.0); b2 = _fdoc("b2", "bank_statement", salary_credit=82000.0)
    assert correlate.check_cross([s1, s2, b1, b2]) == []
    b2.fields[0].value = 70000.0
    assert len(correlate.check_cross([s1, s2, b1, b2])) == 1


def test_declared_income_not_vouched_for_by_a_slip_the_bank_contradicts():
    form = _fdoc("form", "application_form", declared_income=85000.0)
    slip = _fdoc("slip", "salary_slip", net_pay=85000.0, gross=95000.0)       # forged net, bank says 58,000
    bank = _fdoc("bank", "bank_statement", salary_credit=58000.0)
    titles = [f.title for f in correlate.check_cross([form, slip, bank])]
    assert any("Declared income exceeds" in t for t in titles) and any("does not match bank" in t for t in titles)


# ---- semantic categories with confidence -----------------------------------------------------------
def test_category_confidence_is_relative_and_ranked():
    from docforensics.embed_classifier import category_from_sims
    clear = category_from_sims({"other:resume": 0.80, "other:research_paper": 0.57, "salary_slip": 0.50})
    assert clear["key"] == "other:resume" and clear["label"] == "Resume / CV" and clear["confidence"] > 0.95
    close = category_from_sims({"other:financial_report": 0.66, "bank_statement": 0.64, "salary_slip": 0.50})
    assert close["key"] == "other:financial_report" and 0.4 < close["confidence"] < 0.8
    assert [t["key"] for t in close["top"]][:2] == ["other:financial_report", "bank_statement"]


def test_file_category_lets_decisive_pages_outweigh_ambiguous_ones():
    from docforensics.embed_classifier import category_from_pages, describe
    decisive = {"other:financial_report": 0.75, "bank_statement": 0.55, "salary_slip": 0.5}
    vague = [{"other:financial_report": 0.60, "bank_statement": 0.61, "salary_slip": 0.60}] * 4
    cat = category_from_pages([decisive] + vague, [300, 300, 300, 300, 300])
    assert cat["key"] == "other:financial_report" and cat["pages_agree"][1] == 5
    assert "Corporate financial report" in describe(cat) and "pages agree" in describe(cat)


def test_unknown_file_is_described_in_summary_and_finding():
    from docforensics.correlate import check_intra
    from docforensics.llm import template_summary
    d = Document(doc_id="r_p1", path="/x/report.pdf", sha256="0", image_path="x")
    d.doc_type = "unknown"
    d.class_info = {"decision": "only 0 of 10 expected 'bank_statement' fields found (need 3)",
                    "file_category": {"label": "Corporate financial report", "confidence": 0.75, "pages_agree": [3, 3], "key": "k", "top": []}}
    assert "Corporate financial report (confidence 0.75; 3 of 3 pages agree)" in template_summary("c", [d], [])
    assert "most resembles: Corporate financial report" in check_intra([d])[0].explanation
