import json, time, hashlib, os, shutil, sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class AuditTrail:
    """Hash-chained event log of every processing step of ONE run (5.9). Each record carries the SHA-256 of the previous record, so editing,
    removing or reordering events is detectable with verify_chain() / verify_file(). Every record names its run. The log of an earlier run is
    archived under history/ before a case is re-run in the same folder (see archive_previous), and the final digest is printed in the dossier."""

    GENESIS = "0" * 64

    def __init__(self, path, run_id=None):
        self.path = path
        self.run_id = run_id
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.events = []
        self.last = self.GENESIS

    @classmethod
    def resume(cls, path, run_id=None):
        """Continue an existing chained file (the reviewer-decision log keeps growing across runs and sessions)."""
        t = cls(path, run_id)
        t.events = read_events(path)
        t.last = t.events[-1]["hash"] if t.events else cls.GENESIS
        return t

    def log(self, event, **data):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **({"run": self.run_id} if self.run_id else {}), **data, "prev": self.last}
        rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()
        self.last = rec["hash"]
        self.events.append(rec)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def find_hash(self, event, doc_id):
        """Hash of the latest record of this kind for a document: lets a finding point at the events that produced its evidence."""
        for e in reversed(self.events):
            if e.get("event") == event and e.get("doc_id") == doc_id:
                return e["hash"]
        return None

    @property
    def digest(self):
        return self.last


def read_events(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def verify_chain(events):
    """True if every record's hash and back-link are intact."""
    prev = AuditTrail.GENESIS
    for e in events:
        body = {k: v for k, v in e.items() if k != "hash"}
        if e.get("prev") != prev or hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest() != e.get("hash"):
            return False
        prev = e["hash"]
    return True


def verify_file(path):
    """(chain intact, number of records) for a log file as it is on disk."""
    ev = read_events(path)
    return verify_chain(ev), len(ev)


# ---- investigation history: earlier runs of the same case are kept, never overwritten ---------------------------------------------
def new_run_id():
    return time.strftime("%Y%m%dT%H%M%S") + "-" + hashlib.sha256(os.urandom(8)).hexdigest()[:6]


def archive_previous(audit_path, outdir):
    """Copy the existing audit log (if any) to history/<run id>.jsonl before the file is reused. Returns the archived path or None."""
    if not os.path.exists(audit_path) or os.path.getsize(audit_path) == 0:
        return None
    ev = read_events(audit_path)
    rid = next((e.get("run") for e in ev if e.get("run")), None) or "run-" + (ev[0]["ts"].replace(":", "") if ev else "unknown")
    dest_dir = os.path.join(outdir, "history")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{rid}.jsonl")
    if not os.path.exists(dest):
        shutil.copyfile(audit_path, dest)
    return dest


def load_history(outdir):
    p = os.path.join(outdir, "case_history.json")
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return []


def append_history(outdir, entry):
    h = load_history(outdir)
    h.append(entry)
    json.dump(h, open(os.path.join(outdir, "case_history.json"), "w", encoding="utf-8"), indent=2, default=str)
    return h


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "verify":
        ok, n = verify_file(sys.argv[2])
        print(f"{sys.argv[2]}: {n} records, chain {'INTACT' if ok else 'BROKEN'}")
        sys.exit(0 if ok else 1)
    print("usage: python -m docforensics.audit verify <audit.jsonl>")
