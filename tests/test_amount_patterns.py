"""Amount-pattern observations on statements: round-number share, amounts just under common limits, one amount under many descriptions.
Built from fictional table rows; no OCR involved."""
from types import SimpleNamespace as NS
from docforensics.structure import amount_pattern_findings


def _by_file(amounts, descs=None, credit=False):
    rows = []
    for i, a in enumerate(amounts):
        rows.append({"n": i + 1, "role": "txn", "debit": None if credit else a, "credit": a if credit else None,
                     "desc": (descs[i] if descs else f"UPI payment party {i}"), "bbox": [0, i * 10, 100, i * 10 + 8]})
    d = NS(doc_id="doc1", tables=[{"kind": "transactions", "tid": "TB1", "rows": rows}])
    return {"/x/stmt.pdf": [d]}


def _run(by_file):
    n = [0]

    def nid():
        n[0] += 1
        return f"T-{n[0]:03d}"
    return amount_pattern_findings(by_file, nid)


def test_ordinary_amounts_raise_nothing():
    amts = [412.5, 1299.0, 87.25, 2310.4, 640.0, 15.0, 5120.75, 333.33, 780.1, 96.9, 4410.0, 205.55, 1875.2]
    assert _run(_by_file(amts)) == []


def test_mostly_round_amounts_are_one_low_observation():
    f = _run(_by_file([5000, 10000, 7000, 3000, 12000, 8000, 2000, 6000, 9000, 4000, 15000, 1000, 410.5]))
    assert len(f) == 1 and f[0].status == "observed" and f[0].severity == "low" and f[0].confidence < 0.5
    assert "round" in f[0].title.lower() and "not as a result" in f[0].explanation


def test_round_check_needs_enough_rows():
    assert _run(_by_file([5000, 10000, 7000, 3000])) == []


def test_amounts_just_below_a_limit():
    f = _run(_by_file([49500, 49900, 48800, 1250.5]))
    assert len(f) == 1 and "just below 50,000" in f[0].title and f[0].metrics["count"] == 3
    assert "structuring" not in (f[0].title + f[0].explanation).lower() and "fraud" not in (f[0].title + f[0].explanation).lower()


def test_two_near_limit_amounts_are_not_enough():
    assert _run(_by_file([49500, 49900, 1250.5])) == []


def test_same_amount_under_different_descriptions():
    f = _run(_by_file([8200.0] * 4, descs=["Payment to alpha", "Transfer beta", "Cheque gamma", "Cash delta"]))
    assert len(f) == 1 and "same amount 8,200.00" in f[0].title


def test_repeated_fixed_charge_is_quiet():
    assert _run(_by_file([8200.0] * 6, descs=["EMI loan 123"] * 6)) == []
