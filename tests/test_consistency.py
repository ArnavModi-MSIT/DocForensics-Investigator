"""Contextual verification part 2: PAN vs name, date logic, stated age, statement consistency, payslip vs bank. Fabricated data, fixed 'today'."""
import os, sys
from datetime import date
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from docforensics.consistency import consistency_findings, parse_month, parse_period
from docforensics.schema import Document, Field, Word

TODAY = date(2026, 11, 3)


def fld(key, raw, value=None, bbox=(10, 10, 100, 30)):
    return Field(key, key, str(raw), raw if value is None else value, bbox, 0.95)


def doc(doc_id, doc_type, path=None, **fields):
    d = Document(doc_id=doc_id, path=path or doc_id, sha256="0", image_path="")
    d.doc_type = doc_type
    for k, v in fields.items():
        d.fields.append(fld(k, v[0], v[1]) if isinstance(v, tuple) else fld(k, v))
    return d


def titles(docs):
    return [f.title for f in consistency_findings(docs, today=TODAY)]


# ---- PAN vs name ------------------------------------------------------------------------------------
def test_pan_surname_initial_and_holder_type():
    mk = lambda pan, name="Rahul Sharma": doc("a", "application_form", id_number=pan, name=name)
    assert titles([mk("ABCPS7391F")]) == []                                          # P = individual, S = Sharma
    assert titles([mk("ABCPR7391F")]) == []                                          # R = first word: surname-first spelling is accepted
    assert any("surname initial" in t for t in titles([mk("ABCPK7391F")]))
    assert any("holder type" in t for t in titles([mk("ABCCS7391F")]))              # C = company
    assert titles([mk("ABCPK7391F", name="Rahul")]) == []                            # a single word cannot be judged
    assert titles([doc("b", "salary_slip", id_number="ABCCS7391F", name="Rahul Sharma")]) == []   # only forms / ID cards
    assert titles([mk("ABCDE1234F")]) == [] and titles([mk("AAAAA0000A")]) == []                # placeholder / specimen PANs carry no meaning


# ---- date logic ---------------------------------------------------------------------------------------
def test_future_dates_and_reversed_period():
    assert any("Date of birth lies in the future" in t for t in titles([doc("a", "id_card", dob=("01/01/2099", "2099-01-01"))]))
    assert any("Issue date lies in the future" in t for t in titles([doc("a", "id_card", issue_date=("01/01/2099", "2099-01-01"))]))
    assert any("Pay month lies in the future" in t for t in titles([doc("s", "salary_slip", pay_month="December 2099")]))
    assert titles([doc("s", "salary_slip", pay_month="October 2026")]) == []
    assert any("ends before it starts" in t for t in titles([doc("b", "bank_statement", period="31 Oct 26 - 01 Oct 26")]))
    assert any("lies in the future" in t for t in titles([doc("b", "bank_statement", period="01 Dec 2099")]))
    assert titles([doc("b", "bank_statement", period="01 Oct 26 - 31 Oct 26")]) == []


# ---- stated age -------------------------------------------------------------------------------------------
def age_doc(text):
    d = doc("a", "id_card", dob=("14-03-1998", "1998-03-14"))
    d.words = [Word(t, (i * 50, 0, i * 50 + 40, 20), 0.95) for i, t in enumerate(text.split())]
    return d


def test_stated_age_against_date_of_birth():
    assert any("Stated age" in t for t in titles([age_doc("Date of Birth 14-03-1998 Age: 35")]))
    assert titles([age_doc("Date of Birth 14-03-1998 Age: 28")]) == []
    assert titles([age_doc("Date of Birth 14-03-1998 Age: 29")]) == []                  # one year of slack
    assert titles([age_doc("Age as on 01.01.2023: 24")]) == []                           # implied age is taken on the stated date
    assert any("Stated age" in t for t in titles([age_doc("Age as on 01.01.2023: 28")]))
    assert titles([age_doc("Date of Birth / Age: 14-03-1998")]) == []                    # the day of the date is not an age


# ---- bank statement internals ------------------------------------------------------------------------------
def row(n, date_, desc, debit, credit, balance, role="txn"):
    return {"n": n, "role": role, "date": date_, "desc": desc, "debit": debit, "credit": credit, "balance": balance, "cells": {}, "bbox": [0, n * 40, 900, n * 40 + 30],
            "conf": 0.95, "inherited_date": False, "amount": None}


def statement(rows, **fields):
    d = doc("stmt_p1", "bank_statement", path="stmt.pdf", **fields)
    d.tables = [{"tid": "T1", "kind": "transactions", "rows": rows, "bbox": [0, 0, 900, 600], "header": [], "columns": [],
                 "checks": {"rows": len(rows), "checked": len(rows) - 1, "unreadable_rows": 0, "breaks": []}}]
    return d


BASE = [row(1, "2026-10-01", "BALANCE FORWARD", None, None, 50000.0, "opening"), row(2, "2026-10-05", "SALARY NIMBUS PAY TECHNOLOGIES", None, 84000.0, 134000.0),
        row(3, "2026-10-09", "UPI/AMAZON", 1899.0, None, 132101.0)]


def test_genuine_statement_is_silent():
    d = statement(BASE, period="01 Oct 26 - 31 Oct 26", opening_balance=("50,000.00", 50000.0), closing_balance=("132,101.00", 132101.0),
                  salary_credit=("84,000.00", 84000.0))
    assert titles([d]) == []


def test_transactions_outside_the_period_are_flagged():
    rows = BASE + [row(4, "2026-11-02", "UPI/LATE", 100.0, None, 132001.0)]
    assert any("outside the statement period" in t for t in titles([statement(rows, period="01 Oct 26 - 31 Oct 26")]))
    only_date = statement(BASE + [row(4, "2026-10-20", "UPI/X", 10.0, None, 132091.0)], period="16 Oct 2026")        # a single statement date bounds the end only
    assert any("outside the statement period" in t for t in titles([only_date]))


def test_header_balances_must_match_the_table():
    assert any("Opening balance" in t for t in titles([statement(BASE, opening_balance=("48,000.00", 48000.0))]))
    assert any("Closing balance" in t for t in titles([statement(BASE, closing_balance=("133,000.00", 133000.0))]))
    assert any("Salary credit in the header" in t for t in titles([statement(BASE, salary_credit=("85,000.00", 85000.0))]))
    first_txn_only = statement(BASE[1:], opening_balance=("50,000.00", 50000.0))                  # opening derived from the first transaction
    assert titles([first_txn_only]) == []


# ---- payslip vs bank statement -------------------------------------------------------------------------------
def slip(month="October 2026", employer="Nimbus Pay Technologies", net=84000.0):
    return doc("slip_p1", "salary_slip", pay_month=month, employer=employer, net_pay=(f"{net:,.2f}", net))


def test_payslip_month_and_employer_against_the_matching_credit():
    bank = statement(BASE)
    assert titles([slip(), bank]) == []
    assert any("pay month or the month after" in t for t in titles([slip(month="July 2026"), bank]))
    assert titles([slip(month="September 2026"), bank]) == []                                    # paid in the month after
    other = statement([BASE[0], row(2, "2026-10-05", "NEFT FROM ZENITH CORP", None, 84000.0, 134000.0)])
    assert any("does not mention the payslip employer" in t for t in titles([slip(), other]))
    generic = statement([BASE[0], row(2, "2026-10-05", "SALARY CREDIT", None, 84000.0, 134000.0)])
    assert titles([slip(), generic]) == []                                                       # generic narrations cannot be judged
    assert titles([slip(net=90000.0), bank]) == []                                               # amount mismatch belongs to income reconciliation


def test_parsers():
    assert parse_month("October 2026") == (2026, 10) and parse_month("Oct-26") == (2026, 10) and parse_month("10/2026") == (2026, 10)
    assert parse_period("16 Jul 2019") == (None, "2019-07-16") and parse_period("October 2026") == ("2026-10-01", "2026-10-31")
