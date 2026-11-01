"""Calibrate the semantic label matcher on THIS machine's embedding model.
Run:  python tests/calibrate_labels.py        (needs the embedding model; see embed_classifier --check)
Prints, for held-out label paraphrases (not in the dictionary) and for non-label phrases, the best
field key, its similarity and the margin over the runner-up, then suggests DOCFOR_SEM_LABEL_MIN/MARGIN."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from docforensics.embed_classifier import get_embedder
from docforensics.extract import LABELS, EXTRA_LABELS, _label_protos

POS = {  # held-out paraphrases -> expected field key
    "Gross Monthly Pay": "gross", "Total Earnings (Gross)": "gross", "Earnings Total": "gross",
    "Date Of Birth (DOB)": "dob", "Birth Date": "dob", "D.O.B": "dob",
    "Name of Applicant": "name", "Full Legal Name": "name", "Customer Name": "name",
    "Net Salary Payable": "net_pay", "Take-home Amount": "net_pay", "Net Amount Paid": "net_pay",
    "Total Deductions Amount": "deductions", "Deductions (Total)": "deductions",
    "PAN No.": "id_number", "Identity Card Number": "id_number", "Govt ID No": "id_number",
    "A/C Number": "account_no", "Savings Account No": "account_no",
    "Balance Brought Forward": "opening_balance", "Balance Carried Forward": "closing_balance",
    "Salary Deposited": "salary_credit", "Employer Name": "employer", "Company Name": "employer",
    "Residential Address Line": "address", "Date of Issue of Card": "issue_date",
    "Month of Salary": "pay_month", "Statement Duration": "period",
    "Monthly Income Declared": "declared_income",
}
NEG = ["Education", "Technical Skills", "Skills", "Projects", "Experience", "Summary", "Page 1 of 2", "Signature",
       "Branch", "Customer Care", "Terms and Conditions", "Remarks", "Reference Number", "Date", "Total", "Amount",
       "Contact Us", "Declaration", "Authorised Signatory", "Thank you", "Internship", "Certifications"]

emb = get_embedder()
if emb is None:
    sys.exit("No embedding model available (see: python -m docforensics.embed_classifier --check)")
keys, P = _label_protos(emb)


def rank(text):
    v = emb.embed([text])[0]
    per = {}
    for k, s in zip(keys, P @ v):
        per[k] = max(per.get(k, -1), float(s))
    o = sorted(per.items(), key=lambda kv: -kv[1])
    return o[0][0], o[0][1], o[0][1] - o[1][1]


print(f"model: {emb.name}\n\n-- positives (want best key == expected) --")
ps, pm, ok = [], [], 0
for t, exp in POS.items():
    k, s, m = rank(t); hit = k == exp; ok += hit
    if hit: ps.append(s); pm.append(m)
    print(f"{'OK  ' if hit else 'WRONG'} {t:28} -> {k:16} sim {s:.2f} margin {m:.2f}   (want {exp})")
print(f"\n{ok}/{len(POS)} correct")
print("\n-- negatives (should be rejected: low sim or low margin) --")
ns = []
for t in NEG:
    k, s, m = rank(t); ns.append(s)
    print(f"      {t:28} -> {k:16} sim {s:.2f} margin {m:.2f}")
if ps:
    lo, hi = min(ps), max(ns)
    print(f"\nlowest correct positive sim {lo:.2f}; highest negative sim {hi:.2f}")
    if hi < lo:
        print(f"=> clean separation. Suggested: DOCFOR_SEM_LABEL_MIN={(hi + lo) / 2:.2f}  (current default 0.80)")
    else:
        print("=> overlap: sim alone cannot separate them; rely on the margin + type guards, or raise DOCFOR_SEM_LABEL_MIN above "
              f"{hi:.2f} and accept lower recall.")
    print(f"typical margin on correct positives: min {min(pm):.2f} (current DOCFOR_SEM_LABEL_MARGIN default 0.03)")
