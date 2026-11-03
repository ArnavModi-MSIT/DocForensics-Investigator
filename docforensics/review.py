"""Reviewer decisions (5.9 investigation history): what a human concluded about each finding, kept in a hash-chained log next to the case.

The log is separate from the run's audit log because it grows after the run, across sessions. Each decision names the run it refers to and
the digest of that run's audit log (`anchor`), so a decision cannot be silently moved to another run. The system never infers a decision.
"""
import os
from .audit import AuditTrail, read_events, verify_chain

DECISIONS = ("confirmed after checking the original", "dismissed: explained by the original", "needs more information", "note only")
LOG = "review_log.jsonl"


def record_decision(outdir, run_id, run_digest, fid, decision, note="", reviewer=""):
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    trail = AuditTrail.resume(os.path.join(outdir, LOG), run_id)
    return trail.log("review", fid=fid, decision=decision, note=note[:500], reviewer=reviewer[:80], anchor=run_digest)


def load_decisions(outdir, run_id=None):
    ev = [e for e in read_events(os.path.join(outdir, LOG)) if e.get("event") == "review"]
    return [e for e in ev if run_id is None or e.get("run") == run_id]


def verify_decisions(outdir):
    return verify_chain(read_events(os.path.join(outdir, LOG)))
