import json, time, hashlib, os


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class AuditTrail:
    """Hash-chained event log of every processing step for ONE run (5.9). Each record carries the SHA-256 of the
    previous record, so editing, removing or reordering events is detectable with verify_chain(). The log file is
    rewritten when a case is re-run in the same folder, and the final digest is printed in the dossier."""

    GENESIS = "0" * 64

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.events = []
        self.last = self.GENESIS

    def log(self, event, **data):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **data, "prev": self.last}
        rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()
        self.last = rec["hash"]
        self.events.append(rec)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    @property
    def digest(self):
        return self.last


def verify_chain(events):
    """True if every record's hash and back-link are intact."""
    prev = AuditTrail.GENESIS
    for e in events:
        body = {k: v for k, v in e.items() if k != "hash"}
        if e.get("prev") != prev or hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest() != e.get("hash"):
            return False
        prev = e["hash"]
    return True
