"""Fictional documents with real tables and section headings (for structure tests). Everything is invented."""
import os
from PIL import Image, ImageDraw, ImageFont


def _font(sz, bold=False):
    for d in ("/usr/share/fonts/truetype/dejavu", "C:/Windows/Fonts"):
        for n in (("DejaVuSans-Bold.ttf", "arialbd.ttf") if bold else ("DejaVuSans.ttf", "arial.ttf")):
            if os.path.exists(os.path.join(d, n)):
                return ImageFont.truetype(os.path.join(d, n), sz)
    return ImageFont.load_default()


def _right(d, x1, y, text, font, fill=(15, 15, 15)):
    w = d.textlength(text, font=font)
    d.text((x1 - w, y), text, font=font, fill=fill)


def money(x):
    return f"{x:,.2f}"


TXNS = [("03 Oct 26", "ATM WITHDRAWAL CONNAUGHT PLACE", 5000.00, None), ("04 Oct 26", "UPI/SWIGGY/ORDER 8812", 640.50, None),
        ("05 Oct 26", "SALARY NIMBUS PAY TECHNOLOGIES", None, 84000.00), ("07 Oct 26", "NEFT RENT OCTOBER", 22000.00, None),
        ("09 Oct 26", "UPI/AMAZON PAY/PURCHASE", 1899.00, None), ("12 Oct 26", "INTEREST CREDIT", None, 312.75),
        ("15 Oct 26", "ELECTRICITY BILL BSES", 2450.25, None), ("18 Oct 26", "UPI/FRIEND/DINNER SPLIT", None, 750.00),
        ("21 Oct 26", "ATM WITHDRAWAL SAKET", 3000.00, None), ("25 Oct 26", "ONLINE SHOPPING MYNTRA", 4100.00, None)]


def bank_statement(path, opening=48210.40, tamper=None):
    """tamper: None | ('balance', row_no, new_balance) | ('amount', row_no, new_amount) printed without fixing the chain."""
    im = Image.new("RGB", (1240, 1754), (250, 250, 247)); d = ImageDraw.Draw(im); d.ink = d.draw.draw_ink((15, 15, 15))     # default ink is white
    h1, h2, f, fb = _font(40, True), _font(26, True), _font(23), _font(23, True)
    d.text((70, 60), "NIMBUS BANK", font=h1, fill=(10, 50, 110))
    d.text((70, 150), "ACCOUNT DETAILS", font=h2, fill=(10, 10, 10))
    for i, (k, v) in enumerate([("Account Holder", "Rahul Sharma"), ("Account Number", "001234567890"),
                                ("Statement Period", "01 Oct 26 - 31 Oct 26"), ("Currency", "INR")]):
        d.text((70, 205 + i * 38), f"{k}:", font=f, fill=(15, 15, 15)); d.text((340, 205 + i * 38), v, font=f, fill=(15, 15, 15))
    d.text((70, 400), "TRANSACTIONS", font=h2, fill=(10, 10, 10))
    X = dict(date=70, desc=240, wd=800, dep=980, bal=1170)
    y = 460
    d.text((X["date"], y), "Date", font=fb); d.text((X["desc"], y), "Description", font=fb)
    _right(d, X["wd"], y, "Withdrawal", fb); _right(d, X["dep"], y, "Deposit", fb); _right(d, X["bal"], y, "Balance", fb)
    y += 48
    bal = opening
    d.text((X["date"], y), "01 Oct 26", font=f); d.text((X["desc"], y), "BALANCE FORWARD", font=f); _right(d, X["bal"] + 40, y, money(bal), f)
    y += 42
    for n, (dt, desc, wd, dep) in enumerate(TXNS, 1):
        bal = bal - (wd or 0) + (dep or 0)
        pwd, pdep, pbal = wd, dep, bal
        if tamper and tamper[1] == n:
            if tamper[0] == "balance":
                pbal = tamper[2]
            elif tamper[0] == "amount":
                if wd is not None: pwd = tamper[2]
                else: pdep = tamper[2]
        d.text((X["date"], y), dt, font=f); d.text((X["desc"], y), desc, font=f)
        if pwd is not None: _right(d, X["wd"] + 40, y, money(pwd), f)
        if pdep is not None: _right(d, X["dep"] + 40, y, money(pdep), f)
        _right(d, X["bal"] + 40, y, money(pbal), f)
        y += 42
    d.text((70, y + 30), "Closing Balance:", font=fb); d.text((340, y + 30), money(bal), font=fb)
    d.text((70, 1650), "This is a computer generated statement.", font=_font(20), fill=(90, 90, 90))
    im.save(path, "JPEG", quality=92)
    return path


def salary_slip(path, hra=20000, tamper_total=None):
    """Earnings/deductions table; `hra` can be changed without fixing the printed totals to simulate an altered component."""
    im = Image.new("RGB", (1240, 1754), (250, 250, 247)); d = ImageDraw.Draw(im); d.ink = d.draw.draw_ink((15, 15, 15))     # default ink is white
    h1, h2, f, fb = _font(40, True), _font(26, True), _font(23), _font(23, True)
    d.text((70, 60), "SALARY SLIP", font=h1, fill=(10, 50, 110))
    d.text((70, 150), "EMPLOYEE DETAILS", font=h2)
    for i, (k, v) in enumerate([("Employee Name", "Rahul Sharma"), ("Employer", "Nimbus Pay Technologies"), ("Pay Month", "October 2026")]):
        d.text((70, 205 + i * 38), f"{k}:", font=f); d.text((340, 205 + i * 38), v, font=f)
    d.text((70, 380), "EARNINGS AND DEDUCTIONS", font=h2)
    y = 440
    d.text((70, y), "Earnings", font=fb); d.text((470, y), "Amount", font=fb); d.text((640, y), "Deductions", font=fb); d.text((1060, y), "Amount", font=fb)
    ear = [("Basic Salary", 50000), ("House Rent Allowance", hra), ("Special Allowance", 25000)]
    ded = [("Provident Fund", 6000), ("Professional Tax", 200), ("Income Tax", 4800)]
    y += 50
    for (en, ev), (dn, dv) in zip(ear, ded):
        d.text((70, y), en, font=f); _right(d, 570, y, money(ev), f)
        d.text((640, y), dn, font=f); _right(d, 1160, y, money(dv), f)
        y += 42
    d.text((70, y), "Total Earnings", font=fb); _right(d, 570, y, money(tamper_total or 95000), fb)
    d.text((640, y), "Total Deductions", font=fb); _right(d, 1160, y, money(11000), fb)
    d.text((70, y + 70), "Net Pay:", font=fb); d.text((340, y + 70), money(84000), font=fb)
    im.save(path, "JPEG", quality=92)
    return path
