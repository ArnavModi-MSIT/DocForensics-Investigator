"""Embedding-model integrity check: verified, tampered file, unknown revision, nothing cached."""
import hashlib, os
from docforensics.model_integrity import verify_model_dir, PINNED


def _make(tmp_path, content=b"weights", rev="r1"):
    d = tmp_path / "models--x--y" / "snapshots" / rev
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(content)
    return {"models--x--y": {"revision": "r1", "files": {"model.onnx": hashlib.sha256(b"weights").hexdigest()}}}


def test_verified(tmp_path):
    pin = _make(tmp_path)
    assert verify_model_dir(str(tmp_path), pin)["status"] == "verified"


def test_changed_file_is_a_mismatch(tmp_path):
    pin = _make(tmp_path, content=b"tampered")
    r = verify_model_dir(str(tmp_path), pin)
    assert r["status"] == "mismatch" and "model.onnx" in r["detail"]


def test_unknown_revision_is_unpinned_not_trusted_silently(tmp_path):
    pin = _make(tmp_path, rev="other")
    pin["models--x--y"]["revision"] = "r1"
    assert verify_model_dir(str(tmp_path), pin)["status"] == "unpinned"


def test_nothing_cached(tmp_path):
    assert verify_model_dir(str(tmp_path))["status"] == "not found"


def test_recorded_manifest_is_well_formed():
    for spec in PINNED.values():
        assert len(spec["revision"]) == 40 and all(len(h) == 64 for h in spec["files"].values())
