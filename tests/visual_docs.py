"""Fictional documents for the visual-forensics tests: an ID card with a photo block and a QR code, and a form with a signature.
Words/fields are fabricated from the drawing coordinates, so no OCR is needed."""
import io, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from docforensics.schema import Document, Word, Field


def _font(sz, bold=False):
    for d in ("/usr/share/fonts/truetype/dejavu", "C:/Windows/Fonts"):
        for n in (("DejaVuSans-Bold.ttf", "arialbd.ttf") if bold else ("DejaVuSans.ttf", "arial.ttf")):
            if os.path.exists(os.path.join(d, n)):
                return ImageFont.truetype(os.path.join(d, n), sz)
    return ImageFont.load_default()


def jpeg(im, q):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0)
    return Image.open(b).convert("RGB")


def photo_patch(w, h, seed):
    """A continuous-tone 'portrait': smooth gradient, a face-like ellipse and blurred noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = 120 + 60 * (x / w) + 40 * np.sin(y / 23.0)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(img)
    d.ellipse((w * .22, h * .12, w * .78, h * .72), fill=(205, 160, 130))
    d.rectangle((w * .1, h * .72, w * .9, h), fill=(60, 70, 110))
    arr = np.asarray(img, float) + rng.normal(0, 9, (h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))


def id_card(photo_seed=1, forged_photo=False, qr=True):
    """Returns (image, doc). forged_photo: the photo comes from a separately compressed image pasted at an 8x8-misaligned offset."""
    rng = np.random.default_rng(7)
    W, H = 1000, 640
    base = np.full((H, W, 3), (205, 228, 240), np.float32) + rng.normal(0, 2, (H, W, 1))
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    f = _font(30)
    words = []
    for i, (t, x, y) in enumerate([("Government", 300, 30), ("Rahul", 300, 170), ("Sharma", 420, 170), ("DOB:", 300, 230), ("14/03/1994", 400, 230)]):
        d.text((x, y), t, font=f, fill=(15, 15, 15)); bb = d.textbbox((x, y), t, font=f); words.append(Word(t, bb, 0.95))
    if qr:
        q = (rng.random((200, 200)) > 0.5) * 255
        im.paste(Image.fromarray(q.astype(np.uint8)).convert("RGB"), (740, 330))
    page = jpeg(im, 60)
    patch = photo_patch(210, 260, photo_seed)
    if forged_photo:
        patch = jpeg(patch, 50)                                  # its own compression history, own grid origin
        arr = np.asarray(page).copy(); arr[123:123 + 260, 43:43 + 210] = np.asarray(patch); page = Image.fromarray(arr)
    else:
        page.paste(patch, (40, 120)); page = jpeg(page, 60)       # photo goes through the same single compression as the page
    doc = Document(doc_id="card_p1", path="card.png", sha256="0", image_path="")
    doc.doc_type, doc.words, doc.width, doc.height = "id_card", words, W, H
    return page, doc


def signature_form(signature="genuine", label_text="Applicant Signature:"):
    """signature: genuine | none | overlap | pasted. Returns (image, doc) with fabricated words and 6 label:value fields."""
    rng = np.random.default_rng(3)
    W, H = 1240, 1754
    base = np.full((H, W, 3), 246, np.float32) + rng.normal(0, 2.2, (H, W, 1))
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    f = _font(28)
    words, fields = [], []
    rows = [("Applicant Name", "Rahul Sharma"), ("Date of Birth", "14/03/1994"), ("ID Number", "ABCDE1234F"),
            ("Employer", "Nimbus Pay Technologies"), ("Declared Monthly Income", "Rs. 85,000"), ("Address", "Flat 12 Sector 21 Noida")]
    y = 190
    for lab, val in rows:
        d.text((60, y), lab + ":", font=f, fill=(30, 30, 30)); d.text((430, y), val, font=f, fill=(15, 15, 15))
        lb = d.textbbox((60, y), lab + ":", font=f); vb = d.textbbox((430, y), val, font=f)
        words += [Word(lab + ":", lb, 0.95), Word(val, vb, 0.95)]
        fields.append(Field(lab, lab, val, val, vb, 0.95))
        y += 85
    d.text((60, 1320), "I declare that the information given above is true.", font=f, fill=(15, 15, 15))
    words.append(Word("declare", d.textbbox((60, 1320), "I declare that the information given above is true.", font=f), 0.95))
    ly = 1450
    d.line((60, ly - 6, 520, ly - 6), fill=(80, 80, 80), width=2)                 # signature rule
    d.text((60, ly), label_text, font=f, fill=(30, 30, 30))
    lb = d.textbbox((60, ly), label_text, font=f); words.append(Word("Signature:", lb, 0.95))

    def scribble(draw, ox, oy, seed):
        r = np.random.default_rng(seed); x, yy = ox, oy
        for _ in range(14):
            nx, ny = x + r.integers(18, 40), oy + r.integers(-35, 35)
            draw.line((x, yy, (x + nx) // 2, ny + 25, nx, ny), fill=(25, 25, 90), width=3); x, yy = nx, ny
    if signature == "genuine":
        scribble(d, 90, 1405, 1)
    elif signature == "overlap":
        scribble(d, 90, 1335, 2)                                                  # crosses the printed declaration line
    elif signature == "pasted":
        patch = Image.new("RGB", (520, 110), (252, 252, 252)); pd = ImageDraw.Draw(patch); scribble(pd, 30, 55, 1)
        im.paste(patch, (70, 1340))
    im = jpeg(im, 85)
    doc = Document(doc_id="form_p1", path="form.png", sha256="0", image_path="")
    doc.doc_type, doc.words, doc.fields, doc.width, doc.height = "application_form", words, fields, W, H
    return im, doc


def certificate(signature=True, labelled=False):
    """A fictional landscape certificate: border, title, text, a wide handwritten-style signature above a rule with the signatory's
    name and title under it (no 'Signature' label unless labelled=True), a square seal and a QR block."""
    rng = np.random.default_rng(5)
    W, H = 1654, 1170
    base = np.full((H, W, 3), 252, np.float32) + rng.normal(0, 1.0, (H, W, 1))
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    d.rectangle((30, 30, W - 30, H - 30), outline=(20, 90, 160), width=10)                 # decorative border
    f, fb = _font(34), _font(52, True)
    words = []

    def line(text, x, y, font, fill=(20, 110, 60)):
        d.text((x, y), text, font=font, fill=fill)
        words.append(Word(text, d.textbbox((x, y), text, font=font), 0.95))
    line("Certificate of Participation", 420, 330, fb)
    line("This is to certify that Rahul Sharma has participated in the IT Challenge", 300, 460, f)
    line("Chennai", 160, 900, f)
    d.ellipse((1180, 520, 1380, 720), outline=(150, 20, 20), width=6)                      # seal: square, ring-shaped, no OCR words
    d.ellipse((1205, 545, 1355, 695), outline=(150, 20, 20), width=3)
    q = (rng.random((130, 130)) > 0.5) * 255
    im.paste(Image.fromarray(q.astype(np.uint8)).convert("RGB"), (1450, 960))              # QR block
    d.line((620, 905, 1040, 905), fill=(60, 60, 60), width=2)                              # rule under the signature
    line("Meera Kapoor", 700, 925, f)
    line("Head of Programmes", 660, 975, f)
    if labelled:
        line("Signature:", 360, 905, f)
    if signature:
        r = np.random.default_rng(1); x, y = 650, 850
        for _ in range(12):
            nx, ny = x + r.integers(18, 38), 850 + r.integers(-32, 32)
            d.line((x, y, (x + nx) // 2, ny + 22, nx, ny), fill=(25, 25, 90), width=4); x, y = nx, ny
    doc = Document(doc_id="cert_p1", path="cert.png", sha256="0", image_path="")
    doc.doc_type, doc.words, doc.width, doc.height = "unknown", words, W, H
    return jpeg(im, 85), doc
