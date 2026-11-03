"""Cross-document correlation, part 2: case codes, links, unlinked documents, agreement matrix."""
from docforensics.schema import Document, Field, Word
from docforensics.linkage import build_correlation, linkage_findings


def _doc(doc_id, doc_type, code=None, **fields):
    d = Document(doc_id=doc_id, path=doc_id + ".pdf", sha256="0", image_path="", doc_type=doc_type, width=1000, height=1400)
    for k, v in fields.items():
        d.fields.append(Field(k, k, str(v), v, (10, 10 + 20 * len(d.fields), 90, 25 + 20 * len(d.fields)), 1.0))
    if code:
        d.words.append(Word(code, (700, 40, 900, 60), 1.0))
    return d


def _case(code_b="FDX-2026-014/PAY"):
    return [_doc("app", "application_form", "FDX-2026-014/APP", name="Asha Verma", employee_id="NW-3141", account_no="1200457816", declared_income=97000.0),
            _doc("slip", "salary_slip", code_b, name="Asha Verma", employee_id="NW-3141", gross=97000.0, net_pay=89700.0),
            _doc("bank", "bank_statement", "FDX-2026-014/BNK", name="Asha Verma", account_no="1200457816")]


def test_consistent_case_links_everything_and_reports_nothing():
    docs = _case()
    c = build_correlation(docs)
    assert {m["key"]: m["verdict"] for m in c["matrix"]}["name"] == "agree"
    assert {m["key"]: m["verdict"] for m in c["matrix"]}["case_code"] == "agree"
    assert len(c["links"]) == 3 and not c["unlinked"]
    assert linkage_findings(docs, c) == []


def test_different_case_reference_is_reported_with_boxes():
    docs = _case("FDX-2026-099/PAY")
    c = build_correlation(docs)
    f = linkage_findings(docs, c)
    assert len(f) == 1 and f[0].title.startswith("Case reference differs") and f[0].status == "inconsistency"
    assert all(r["bbox"] for r in f[0].refs)


def test_document_sharing_nothing_is_flagged_only_when_others_are_linked():
    docs = _case() + [_doc("other", "bank_statement", name="Ravi Menon", account_no="9988776655", dob="1970-01-01")]
    docs[-1].fields.append(Field("employee_id", "e", "ZZ-1", "ZZ-1", (1, 1, 2, 2), 1.0))
    c = build_correlation(docs)
    assert c["unlinked"] == ["other"]
    f = [x for x in linkage_findings(docs, c) if "shares no identifier" in x.title]
    assert f and f[0].severity == "medium"
    lonely = [_doc("a", "salary_slip", name="X One"), _doc("b", "bank_statement", name="Y Two")]
    assert build_correlation(lonely)["unlinked"] == []          # nothing is linked at all: no document is singled out


def test_matrix_amounts_and_payroll_match():
    docs = _case()
    docs[2].facts["payroll_credits"] = [{"amount": 89700.0, "date": "2026-10-30", "desc": "Payroll", "payer": None, "bbox": [1, 1, 2, 2], "raw": "89,700"}]
    rows = {m["field"]: m for m in build_correlation(docs)["matrix"]}
    assert rows["Payslip net pay vs bank payroll credit"]["verdict"] == "agree"
    docs[2].facts["payroll_credits"][0]["amount"] = 99700.0
    rows = {m["field"]: m for m in build_correlation(docs)["matrix"]}
    assert rows["Payslip net pay vs bank payroll credit"]["verdict"] == "differ"
