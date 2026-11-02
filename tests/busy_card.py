"""Fictional Voter-ID-style card on a busy tricolour / guilloche background (for OCR-engine tests).
`harsh` in [0, 1]: 0 = faint pattern, 1 = very dense pattern + noise + heavy JPEG."""
import os, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont


def _font(sz, bold=False):
    for d in ("/usr/share/fonts/truetype/dejavu", "C:/Windows/Fonts"):
        for n in (("DejaVuSans-Bold.ttf", "arialbd.ttf") if bold else ("DejaVuSans.ttf", "arial.ttf")):
            if os.path.exists(os.path.join(d, n)):
                return ImageFont.truetype(os.path.join(d, n), sz)
    return ImageFont.load_default()


def busy_card(path, harsh=0.5, seed=3, name="Rahul Kumar Sharma", father="Suresh Sharma", dob="14-03-1998",
              epic="ABC1234567"):
    rng = np.random.default_rng(seed)
    W, H = 1000, 640
    bg = np.zeros((H, W, 3), np.float32)
    sa, wh, gr = np.array([90, 150, 240]), np.array([225, 235, 235]), np.array([120, 190, 130])
    for i in range(H):
        f = i / H
        bg[i] = sa * (1 - f / 0.33) + wh * (f / 0.33) if f < 0.33 else (
            wh if f < 0.66 else wh * (1 - (f - 0.66) / 0.34) + gr * ((f - 0.66) / 0.34))
    xs = np.arange(W)
    dark = 0.9 - 0.35 * harsh                              # wave darkness multiplier
    for k in range(int(40 + 100 * harsh)):
        ys = (H / 2 + (H / 2.2) * np.sin(xs / 22.0 + k * 0.3) * np.cos(k * 0.13)).astype(int)
        for xx, yy in zip(xs, ys):
            if 0 <= yy < H:
                bg[max(0, yy - 1):yy + 2, xx] *= dark
    for k in range(int(40 * harsh)):                       # fine diagonal security hatching
        cv2.line(bg, (0, k * 18), (W, k * 18 - 300), (110, 110, 130), 1)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    im = Image.fromarray(bg[:, :, ::-1].copy()); d = ImageDraw.Draw(im)
    f, f2, tc = _font(26, True), _font(24), (40, 35, 60)
    d.text((60, 40), "ELECTION COMMISSION OF INDIA", fill=tc, font=f)
    d.text((60, 90), "IDENTITY CARD", fill=tc, font=f)
    d.text((60, 200), f"Elector's Name : {name}", fill=tc, font=f2)
    d.text((60, 250), f"Father's Name : {father}", fill=tc, font=f2)
    d.text((60, 300), "Sex : Male", fill=tc, font=f2)
    d.text((60, 350), f"Date of Birth : {dob}", fill=tc, font=f2)
    d.text((60, 520), f"EPIC No. {epic}", fill=tc, font=f)
    a = cv2.GaussianBlur(np.array(im), (3, 3), 0.5 + 0.5 * harsh).astype(np.float32) + rng.normal(0, 3 + 7 * harsh, (H, W, 3))
    cv2.imwrite(path, np.clip(a, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(85 - 30 * harsh)])
    return path
