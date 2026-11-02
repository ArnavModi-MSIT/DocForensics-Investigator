"""End-to-end test with FICTIONAL Aadhaar-style cards (front+back): unlabelled name/number, bilingual-style layout,
multi-line address with a QR block. Builds its own images. Run: python tests/test_id_cards.py"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PIL import Image, ImageDraw, ImageFont
from docforensics.id_patterns import verhoeff_check_digit
from docforensics.pipeline import run_case

OUT = os.path.join(os.path.dirname(__file__), "..", "output", "_t_id")


def font(sz, bold=False):
    for d in ("/usr/share/fonts/truetype/dejavu", "C:/Windows/Fonts"):
        for n in (("DejaVuSans-Bold.ttf", "arialbd.ttf") if bold else ("DejaVuSans.ttf", "arial.ttf")):
            if os.path.exists(os.path.join(d, n)):
                return ImageFont.truetype(os.path.join(d, n), sz)
    return ImageFont.load_default()


def card(path, side, number, seed):
    rng = np.random.default_rng(seed)
    a = np.full((640, 1000, 3), (205, 228, 240), np.float32) + rng.normal(0, 2, (640, 1000, 1))
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    if side == "front":
        d.text((300, 30), "Government of India", font=font(34, True), fill=(10, 60, 30))
        d.text((300, 170), "Rohan Mehta", font=font(34), fill=(15, 15, 15))
        d.text((300, 230), "DOB: 14/03/1994", font=font(32), fill=(15, 15, 15))
        d.text((300, 290), "Male", font=font(32), fill=(15, 15, 15))
        d.text((200, 470), number, font=font(60, True), fill=(10, 10, 10))
        d.text((90, 570), "Aadhaar is a proof of identity, not of citizenship", font=font(24), fill=(60, 20, 20))
        d.rectangle((40, 120, 250, 380), outline=(90, 90, 90), width=3)          # photo box
    else:
        d.text((300, 30), "Unique Identification Authority of India", font=font(30, True), fill=(10, 60, 30))
        d.text((60, 160), "Address: C/O Mohan Mehta, Flat 12,", font=font(30), fill=(15, 15, 15))
        d.text((60, 205), "Sector 21, Noida,", font=font(30), fill=(15, 15, 15))
        d.text((60, 250), "Gautam Buddha Nagar, Uttar Pradesh, 201301", font=font(30), fill=(15, 15, 15))
        qr = (rng.random((200, 200)) > 0.5) * 255
        im.paste(Image.fromarray(qr.astype(np.uint8)).convert("RGB"), (740, 130))
        d.text((200, 470), number, font=font(60, True), fill=(10, 10, 10))
    im.save(path, "JPEG", quality=90)


def run(tag, number):
    os.makedirs(os.path.join(OUT, tag), exist_ok=True)
    paths = []
    for side in ("front", "back"):
        p = os.path.join(OUT, tag, f"{side}.jpg"); card(p, side, number, 3 if side == "front" else 4); paths.append(p)
    r = run_case(paths, os.path.join(OUT, tag + "_out"))
    print(f"\n[{tag}]")
    for d in r["docs"]:
        print(f"  {d.doc_id:6} type={d.doc_type:9} fields={[(f.key, f.match) for f in d.fields]} checksum_ok={d.facts.get('aadhaar_checksum_ok')}")
        print(f"         why: {d.class_info['decision'][:100]}")
    print("  findings:", [f.title for f in r["findings"]])
    return r


if __name__ == "__main__":
    base = "29834756102"
    good = base + str(verhoeff_check_digit(base))
    good_fmt = f"{good[:4]} {good[4:8]} {good[8:]}"
    bad = good[:-1] + str((int(good[-1]) + 1) % 10)
    bad_fmt = f"{bad[:4]} {bad[4:8]} {bad[8:]}"
    v = run("valid", good_fmt)
    c = run("corrupted", bad_fmt)
    fails = []
    for tag, r in (("valid", v), ("corrupted", c)):
        if [d.doc_type for d in r["docs"]] != ["id_card", "id_card"]:
            fails.append(f"{tag}: pages not both recognised as id_card")
        front, back = r["docs"]
        for key in ("name", "dob", "aadhaar_no"):
            if not front.get(key):
                fails.append(f"{tag}: front missing {key}")
        for key in ("address", "aadhaar_no"):
            if not back.get(key):
                fails.append(f"{tag}: back missing {key}")
        blob = open(r["json"], encoding="utf-8").read()
        for digits in (good, bad):
            if digits in blob.replace(" ", ""):
                fails.append(f"{tag}: full Aadhaar digits leaked into dossier.json")
    if any("checksum" in f.title for f in v["findings"]):
        fails.append("valid: unexpected checksum finding")
    if sum("checksum" in f.title for f in c["findings"]) != 2:
        fails.append("corrupted: expected a checksum finding on both pages")
    print("\nID-CARD TEST:", "PASSED" if not fails else "FAILED")
    for f in fails:
        print("   FAIL:", f)
    sys.exit(1 if fails else 0)
