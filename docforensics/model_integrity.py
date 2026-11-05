"""Integrity check of the downloaded embedding model (supply-chain hygiene, OWASP ML06 / ML10 in spirit).

The semantic classifier downloads a model on first use. The SHA-256 of each file of the revision we tested is recorded below. These values were
computed from the files as downloaded to our development machine ('trust on first use'); they were not compared with a publisher-signed value, so
they detect a changed or corrupted file, not a malicious revision that was already in place when they were recorded. A file that differs from its
recorded hash makes the model unavailable (the classifier falls back to keywords and structure); a different, unrecorded revision is used but
reported as 'unpinned'. DOCFOR_MODEL_VERIFY=off skips the check.
"""
import hashlib, os

PINNED = {
    "models--Qdrant--bge-small-en-v1.5-onnx-Q": {
        "revision": "aa8f8b060edb00e03bfdd08813a2949946c8ba55",
        "files": {
            "model_optimized.onnx": "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431",
            "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            "config.json": "13582bcf2effc85b7bf3d3f5532e686bc1c9ce86bb009d10f0ec33cbe92299dd",
            "special_tokens_map.json": "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
            "tokenizer_config.json": "0b29c7bfc889e53b36d9dd3e686dd4300f6525110eaa98c76a5dafceb2029f53",
        },
    }
}


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_dir(model_dir, pinned=None):
    """{'status': 'verified' | 'mismatch' | 'unpinned' | 'not found', 'detail': str, 'files': {name: sha256}} for the cached model snapshots under model_dir."""
    pinned = PINNED if pinned is None else pinned
    found = False
    for repo, spec in pinned.items():
        snaps = os.path.join(model_dir, repo, "snapshots")
        if not os.path.isdir(snaps):
            continue
        found = True
        revs = sorted(os.listdir(snaps))
        if spec["revision"] not in revs:
            return {"status": "unpinned", "detail": f"revision {revs[-1][:12] if revs else '?'} of {repo} is not in the recorded manifest", "files": {}}
        base, seen, bad = os.path.join(snaps, spec["revision"]), {}, []
        for name, want in spec["files"].items():
            p = os.path.join(base, name)
            if not os.path.exists(p):
                bad.append(f"{name} missing")
                continue
            seen[name] = _sha(p)
            if seen[name] != want:
                bad.append(f"{name} differs from its recorded SHA-256")
        if bad:
            return {"status": "mismatch", "detail": "; ".join(bad), "files": seen}
        return {"status": "verified", "detail": f"{len(seen)} files match the recorded SHA-256 values (revision {spec['revision'][:12]})", "files": seen}
    return {"status": "not found", "detail": "no recorded model snapshot found under " + model_dir, "files": {}}
