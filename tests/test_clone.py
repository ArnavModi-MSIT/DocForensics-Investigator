"""Duplicated-region and image-metadata checks on synthetic images."""

import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from docforensics.forensics_clone import find_clones, read_image_meta, image_meta_findings
from docforensics.schema import Document


def _font(n):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", n)
    except OSError:
        return ImageFont.load_default()


def _page(seed=1, words=14, w=1240, h=900):
    """A noisy scan-like page with lines of text; every word is drawn, then independent noise is added at the end."""
    rng = np.random.default_rng(seed)
    im = Image.new("L", (w, h), 238)
    d = ImageDraw.Draw(im)
    f = _font(30)
    for i in range(words):
        d.text((60 + (i % 3) * 380, 80 + (i // 3) * 120), ["Amount", "Balance", "Account 4471", "Total 98,500", "Employee"][i % 5] + f" {i * 37 % 90 + 10}", font=f, fill=30)
    a = np.asarray(im, np.float32) + rng.normal(0, 4, (h, w))
    return np.clip(a, 0, 255)


def test_copied_region_is_found():
    a = _page()
    a[600:665, 700:1020] = a[75:140, 60:380]                       # paste a 320x65 area holding text elsewhere (same pixels, same noise)
    found = find_clones(a.astype(np.uint8))
    assert found and found[0]["ncc"] >= 0.55
    sx, sy = found[0]["shift"]
    assert abs(sx - 640) <= 6 and abs(sy - 525) <= 6                # an off-grid shift (525 is not a multiple of the block stride) is still found
    b = io.BytesIO()
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(b, "JPEG", quality=95)
    assert find_clones(np.asarray(Image.open(b)))                   # light compression after the paste keeps enough of the noise


def test_repeated_text_with_independent_noise_is_not_a_clone():
    rng = np.random.default_rng(5)
    im = Image.new("L", (1240, 900), 238)
    d = ImageDraw.Draw(im)
    f = _font(30)
    for i in range(12):
        d.text((60 + (i % 3) * 380, 80 + (i // 3) * 120), "Total 98,500", font=f, fill=30)         # the same word twelve times
    a = np.clip(np.asarray(im, np.float32) + rng.normal(0, 4, (900, 1240)), 0, 255).astype(np.uint8)
    assert find_clones(a) == []


def test_clean_page_without_repetition_is_clean():
    assert find_clones(_page(seed=3).astype(np.uint8)) == []


def test_image_metadata_editor_and_modified_after_capture(tmp_path):
    im = Image.new("RGB", (64, 64), "white")
    ex = Image.Exif()
    ex[0x0131] = "Adobe Photoshop 25.0"
    ex[0x0132] = "2026:10:30 15:00:00"
    ex.get_ifd(0x8769)[0x9003] = "2026:10:30 09:00:00"
    p = tmp_path / "a.jpg"
    im.save(p, exif=ex)
    d = Document(doc_id="a", path=str(p), sha256="0", image_path=str(p))
    f = image_meta_findings(d, 1)
    assert f and "editor" in f[0].explanation and "later than the capture" in f[0].explanation and f[0].status == "observed"
    q = tmp_path / "b.jpg"
    im.save(q)                                                       # no metadata: nothing to report
    assert image_meta_findings(Document(doc_id="b", path=str(q), sha256="0", image_path=str(q)), 1) == []
