"""End-to-end gates (OCR required, ~1 min): a clean and a tampered baseline case through the whole pipeline."""
import glob, json, os, subprocess, sys
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _run(*args):
    env = dict(os.environ, DOCFOR_EMBED=os.environ.get("DOCFOR_EMBED", "off"), PYTHONPATH=ROOT)
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=170)


@pytest.fixture(scope="module")
def baseline_samples():
    """samples/ is not stored in git (fictional, regenerated deterministically): create it on a fresh clone."""
    need = [os.path.join(ROOT, "samples", c, "truth.json") for c in ("case_clean", "case_tampered")]
    if not all(os.path.exists(p) for p in need):
        r = _run("synth/generate.py")
        assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]


def test_baseline_eval_gate(baseline_samples):
    r = _run("tests/eval_synth.py", "case_clean", "case_tampered")
    assert r.returncode == 0, r.stdout[-1500:]


def test_id_card_gate():
    r = _run("tests/test_id_cards.py")
    assert r.returncode == 0, r.stdout[-1500:]
