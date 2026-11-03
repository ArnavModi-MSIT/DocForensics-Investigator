"""Audit trail (5.9): chain verification on disk, archived history, finding records with evidence and lineage, reviewer decisions."""
import json, os
import pytest
from docforensics.audit import AuditTrail, verify_chain, verify_file, read_events, load_history
from docforensics.pipeline import run_case
from docforensics.review import record_decision, load_decisions, verify_decisions
from tests.test_pdf_layer import _page


def test_chain_detects_edit_removal_and_reorder(tmp_path):
    p = str(tmp_path / "a.jsonl")
    t = AuditTrail(p, "r1")
    for i in range(4):
        t.log("step", n=i)
    assert verify_file(p) == (True, 4)
    ev = read_events(p)
    bad = [dict(e) for e in ev]; bad[1]["n"] = 99
    assert not verify_chain(bad)
    assert not verify_chain(ev[:1] + ev[2:])
    assert not verify_chain([ev[1], ev[0]] + ev[2:])


def test_run_records_findings_with_evidence_and_keeps_history(tmp_path):
    pdf = tmp_path / "slip.pdf"
    _page(str(pdf), net="INR 99,700", patch=True)
    out = str(tmp_path / "case")
    r1 = run_case([str(pdf)], out, case_id="c")
    r2 = run_case([str(pdf)], out, case_id="c")
    assert r1["run_id"] != r2["run_id"]
    hist = load_history(out)
    assert [h["run_id"] for h in hist] == [r1["run_id"], r2["run_id"]] and all(h["chain_ok"] for h in hist)
    assert os.path.exists(os.path.join(out, "history", r1["run_id"] + ".jsonl"))              # the first run's log survived the re-run
    assert verify_file(os.path.join(out, "history", r1["run_id"] + ".jsonl"))[0]
    assert r2["dossier"]["audit"]["chain_ok"] and len(r2["dossier"]["audit"]["earlier_runs"]) == 1
    ev = read_events(os.path.join(out, "audit.jsonl"))
    fe = [e for e in ev if e["event"] == "finding" and e["kind"] == "visual"][0]
    assert fe["refs"][0]["bbox"] and fe["source_sha256"]["slip"] == r2["docs"][0].sha256 and fe["evidence_sha256"]
    assert fe["produced_by"]["slip"]["ingest"] and fe["produced_by"]["slip"]["fields_final"]
    hashes = {e["hash"] for e in ev}
    assert fe["produced_by"]["slip"]["extract"] in hashes                                       # lineage points at real events of this run
    assert any(e["event"] == "fields_final" and any(f["key"] == "net_pay" for f in e["fields"]) for e in ev)


def test_review_log_is_chained_and_anchored(tmp_path):
    out = str(tmp_path)
    e1 = record_decision(out, "run1", "d" * 64, "V-001", "needs more information", "ask issuer", "A. Reviewer")
    e2 = record_decision(out, "run1", "d" * 64, "X-001", "note only")
    assert e2["prev"] == e1["hash"] and e1["anchor"] == "d" * 64 and verify_decisions(out)
    assert [e["fid"] for e in load_decisions(out, "run1")] == ["V-001", "X-001"] and load_decisions(out, "other") == []
    with pytest.raises(ValueError):
        record_decision(out, "run1", "d" * 64, "V-001", "approved as genuine")                   # only human review outcomes are accepted
    p = os.path.join(out, "review_log.jsonl")
    lines = open(p).read().splitlines()
    rec = json.loads(lines[0]); rec["decision"] = "note only"
    open(p, "w").write("\n".join([json.dumps(rec)] + lines[1:]) + "\n")
    assert not verify_decisions(out)
