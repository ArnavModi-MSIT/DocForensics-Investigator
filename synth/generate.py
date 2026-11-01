"""Generate synthetic SPECIMEN KYC/loan document sets with known tampering and capture degradations.

All people, numbers and companies are fictional. Output: samples/<case>/*.jpg + truth.json + fields_truth.json
Cases: case_clean, case_tampered (clean digital renders) and, per capture profile, case_<profile>_clean /
case_<profile>_tampered (the same pages after rotation, perspective, blur, noise, lighting, heavy JPEG).
Usage: python synth/generate.py
"""
import io, json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1000, 1300
OUT = os.path.join(os.path.dirname(__file__), "..", "samples")
FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "C:/Windows/Fonts", "/Library/Fonts",
             "/System/Library/Fonts/Supplemental"]

# capture-degradation profiles applied to the WHOLE page after any tampering (a photo/scan of a forged page)
PROFILES = {
    "scan":  dict(rot=1.5, persp=0.0,   blur=0.5, noise=2.0, jpeg=70, grad=0,  bright=0),
    "phone": dict(rot=3.0, persp=0.012, blur=0.9, noise=1.5, jpeg=55, grad=18, bright=-10),
}
LABEL_KEY = {"Applicant Name": "name", "Name": "name", "Employee Name": "name", "Account Holder": "name",
             "Date of Birth": "dob", "ID Number": "id_number", "Address": "address", "Employer": "employer",
             "Declared Monthly Income": "declared_income", "Issue Date": "issue_date", "Pay Month": "pay_month",
             "Gross Earnings": "gross", "Total Deductions": "deductions", "Net Pay": "net_pay",
             "Account Number": "account_no", "Statement Period": "period", "Opening Balance": "opening_balance",
             "Salary Credit": "salary_credit", "Closing Balance": "closing_balance"}


def font(names, size):
    for d in FONT_DIRS:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


SANS = ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf"]
SERIF = ["DejaVuSerif.ttf", "times.ttf", "Times New Roman.ttf"]
BOLD = ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"]


def money(x):
    return f"{int(x):,}"


PERSON = dict(name="Rohan Mehta", dob="14/03/1994", pid="ABCDE1234F",
              addr="Flat 12, Sector 21, Noida UP 201301", employer="Nimbus Analytics Pvt Ltd",
              gross=95000, ded=10000, acct="50100234567891")


def base_canvas(seed):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 246, dtype=np.float32)
    img += rng.normal(0, 2.2, (H, W, 1))            # paper grain
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def layout(title, rows, seed):
    """rows: list of (label, value). Returns (image, {label: bbox_of_value}, {label: value})."""
    im = base_canvas(seed)
    d = ImageDraw.Draw(im)
    d.text((60, 70), title, font=font(BOLD, 36), fill=(20, 20, 20))
    d.line((60, 130, W - 60, 130), fill=(60, 60, 60), width=2)
    fl, fv = font(SANS, 28), font(SANS, 28)
    boxes, vals, y = {}, {}, 190
    for label, value in rows:
        d.text((60, y), f"{label}:", font=fl, fill=(30, 30, 30))
        d.text((430, y), value, font=fv, fill=(15, 15, 15))
        boxes[label] = d.textbbox((430, y), value, font=fv)
        vals[label] = value
        y += 85
    d.text((60, H - 90), "SPECIMEN - FICTIONAL DATA - FOR HACKATHON TESTING ONLY",
           font=font(SANS, 18), fill=(120, 120, 120))
    return im, boxes, vals


def jpeg(im, q):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0)
    return Image.open(b).convert("RGB")


def patch(im, bbox, new_text, size=28, serif=True):
    """Paste-over tampering: new value rendered elsewhere (other font, other 'scan'
    with its own paper tone, grain, ink darkness and resampling blur) and pasted in."""
    from PIL import ImageFilter
    x0, y0, x1, y1 = bbox
    pad = 10
    pw, ph = (x1 - x0) + 2 * pad + 40, (y1 - y0) + 2 * pad
    rng = np.random.default_rng(sum(map(ord, new_text)))
    tone = 246 + 3.5                                  # slightly different paper tone
    arr = np.full((ph, pw, 3), tone, dtype=np.float32) + rng.normal(0, 1.1, (ph, pw, 1))
    pimg = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    f = font(SERIF if serif else SANS, size)
    ImageDraw.Draw(pimg).text((pad, pad - 2), new_text, font=f, fill=(48, 48, 48))
    pimg = pimg.resize((int(pw * 1.08), int(ph * 1.08)), Image.BICUBIC).resize((pw, ph), Image.BICUBIC)
    pimg = pimg.filter(ImageFilter.GaussianBlur(0.6))
    im.paste(pimg, (x0 - pad, y0 - pad))
    return (x0, y0, x0 + pw - 2 * pad, y1)


def degrade(im, P, seed):
    """Simulate a scan / phone photo of the page: lighting, rotation, perspective, blur, noise."""
    import cv2
    rng = np.random.default_rng(seed)
    a = np.asarray(im).astype(np.float32)
    h, w = a.shape[:2]
    if P["grad"]:
        gx = np.linspace(-1, 1, w)[None, :, None]; gy = np.linspace(-1, 1, h)[:, None, None]
        t = rng.uniform(0, 2 * np.pi)
        a += P["grad"] * (np.cos(t) * gx + np.sin(t) * gy)
        a *= 1 - 0.12 * ((gx ** 2 + gy ** 2) / 2)               # vignette
    a = np.clip(a + P["bright"], 0, 255).astype(np.uint8)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-P["rot"], P["rot"]), 1.0)
    a = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(225, 225, 225))
    if P["persp"]:
        d = P["persp"] * min(w, h)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + rng.uniform(-d, d, (4, 2)).astype(np.float32)
        a = cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, dst), (w, h),
                                flags=cv2.INTER_CUBIC, borderValue=(225, 225, 225))
    if P["blur"]:
        a = cv2.GaussianBlur(a, (0, 0), P["blur"])
    if P["noise"]:
        a = np.clip(a.astype(np.float32) + rng.normal(0, P["noise"], a.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def docs(p, seed, variant):
    P = dict(p); P.update(variant.get("shown", {}))
    net = P["gross"] - P["ded"]
    sets = {}
    sets["application_form"] = layout("LOAN APPLICATION FORM", [
        ("Applicant Name", P["name"]), ("Date of Birth", P["dob"]),
        ("ID Number", P["pid"]), ("Address", P["addr"]), ("Employer", P["employer"]),
        ("Declared Monthly Income", "Rs. " + money(variant.get("declared", net)))], seed)
    sets["id_card"] = layout("NATIONAL IDENTITY CARD", [
        ("Name", P["name"]), ("Date of Birth", variant.get("id_dob", P["dob"])),
        ("ID Number", variant.get("id_pid", P["pid"])), ("Address", P["addr"]),
        ("Issue Date", "02/06/2019")], seed + 1)
    sets["salary_slip"] = layout("SALARY SLIP", [
        ("Employee Name", P["name"]), ("Employer", P["employer"]), ("Pay Month", "October 2026"),
        ("Gross Earnings", "Rs. " + money(P["gross"])),
        ("Total Deductions", "Rs. " + money(variant.get("slip_ded", P["ded"]))),
        ("Net Pay", "Rs. " + money(variant.get("slip_net", net)))], seed + 2)
    sets["bank_statement"] = layout("BANK ACCOUNT STATEMENT", [
        ("Account Holder", variant.get("bank_name", P["name"])), ("Account Number", P["acct"]),
        ("Statement Period", "01/10/2026 - 31/10/2026"), ("Opening Balance", "Rs. 120,000"),
        ("Salary Credit", "Rs. " + money(variant.get("bank_credit", net))),
        ("Closing Balance", "Rs. " + money(120000 + variant.get("bank_credit", net) - 31500))], seed + 3)
    return sets


def build_case(name, variant, tamper, profile=None):
    """tamper: list of (doc, label, new_text) pasted over after a first compression."""
    outdir = os.path.join(OUT, name); os.makedirs(outdir, exist_ok=True)
    seed0 = sum(map(ord, name)) % 1000
    sets = docs(PERSON, seed0, variant)
    truth, ftruth = [], {}
    for i, (doc, (im, boxes, vals)) in enumerate(sets.items()):
        t = [x for x in tamper if x[0] == doc]
        if t:
            im = jpeg(im, 75)                       # prior compression history
            for _, label, new_text in t:
                nb = patch(im, boxes[label], new_text)
                vals[label] = new_text
                truth.append(dict(doc=doc, label=label, bbox=list(nb), kind="visual_patch"))
        ftruth[doc] = {LABEL_KEY[l]: v for l, v in vals.items()}
        q = 92
        if profile:
            P = PROFILES[profile]
            im = degrade(im, P, seed0 * 10 + i)
            q = P["jpeg"]
        im.save(os.path.join(outdir, f"{doc}.jpg"), "JPEG", quality=q)
    for k, v in variant.get("content_truth", {}).items():
        truth.append(dict(doc=k, label=v, kind="content_mismatch"))
    json.dump(truth, open(os.path.join(outdir, "truth.json"), "w"), indent=2)
    json.dump(ftruth, open(os.path.join(outdir, "fields_truth.json"), "w"), indent=2)
    print("wrote", name, f"({len(truth)} ground-truth issues, profile={profile or 'digital'})")


TAMPER_VARIANT = dict(
    slip_ded=37000, slip_net=58000, bank_credit=58000,   # true net was 58,000
    bank_name="Rohan Mehra",                              # content-only mismatch
    id_pid="ABCDE1284F",                                  # content-only mismatch
    declared=85000,                                       # inflated income claim
    content_truth={"bank_statement": "Account Holder", "id_card": "ID Number",
                   "application_form": "Declared Monthly Income"})
TAMPER_PATCHES = [("salary_slip", "Net Pay", "Rs. 85,000"),        # patched: 58,000 -> 85,000
                  ("id_card", "Date of Birth", "14/03/1990")]      # patched DOB

if __name__ == "__main__":
    for prof in (None, *PROFILES):
        pre = "case_" + (prof + "_" if prof else "")
        build_case(pre + "clean", {}, [], prof)
        build_case(pre + "tampered", TAMPER_VARIANT, TAMPER_PATCHES, prof)
