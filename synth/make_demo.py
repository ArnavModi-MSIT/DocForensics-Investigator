"""Build the small, committed demonstration dataset in demo_data/ (all people, numbers and companies are fictional).

  demo_data/case_clean            4 scanned-style pages, internally consistent          (expect: no substantive finding)
  demo_data/case_tampered         same case with planted edits (DOB patch, net-pay patch, holder name mismatch, ...)
  demo_data/case_scan_tampered    the tampered case after a scan-like degradation       (expect: findings, lower confidence)
  demo_data/pdf_payslip_clean     born-digital PDF payslip                              (expect: no substantive finding)
  demo_data/pdf_payslip_patched   the same payslip with the net pay overpainted         (expect: potential alteration + arithmetic)

Each image case carries truth.json (where the edits were planted). Usage: python synth/make_demo.py
"""
import os, shutil, subprocess, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SAMPLES = os.path.join(ROOT, "samples")
DEMO = os.path.join(ROOT, "demo_data")


def _payslip(path, patch):
    import pymupdf
    doc = pymupdf.open()
    pg = doc.new_page(width=595, height=842)
    pg.draw_rect(pymupdf.Rect(0, 0, 595, 100), fill=(0.08, 0.15, 0.24), color=None)
    pg.insert_text((43, 45), "NORTHWIND DEMO LABS", fontname="hebo", fontsize=14, color=(1, 1, 1))
    pg.insert_text((43, 72), "Salary Statement | October 2026", fontname="helv", fontsize=12, color=(1, 1, 1))

    def pair(x, y, label, value):
        pg.insert_text((x, y), label, fontname="hebo", fontsize=7.4, color=(0.4, 0.45, 0.5))
        pg.insert_text((x, y + 20), value, fontname="helv", fontsize=10, color=(0.05, 0.1, 0.15))
    pair(54, 160, "EMPLOYEE NAME", "Asha Verma")
    pair(326, 160, "EMPLOYEE ID", "NW-3141")
    pair(54, 220, "PAYROLL PERIOD", "October 2026")
    pair(326, 220, "PAYMENT DATE", "30 Oct 2026")
    for i, (a, b) in enumerate([("Base pay", "78,000"), ("Allowance", "12,000"), ("Other", "7,000")]):
        y = 330 + 28 * i
        pg.insert_text((58, y), a, fontname="helv", fontsize=9.7)
        pg.insert_text((243, y), b, fontname="helv", fontsize=9.7)
    for i, (a, b) in enumerate([("Retirement", "4,200"), ("Income tax", "3,100")]):
        y = 330 + 28 * i
        pg.insert_text((328, y), a, fontname="helv", fontsize=9.7)
        pg.insert_text((508, y), b, fontname="helv", fontsize=9.7)
    for x, t in [(58, "EARNINGS"), (260, "INR"), (328, "DEDUCTIONS"), (519, "INR")]:
        pg.insert_text((x, 300), t, fontname="hebo", fontsize=8.6)
    pair(65, 440, "GROSS PAY", "INR 97,000")
    pair(265, 440, "TOTAL DEDUCTIONS", "INR 7,300")
    pg.draw_rect(pymupdf.Rect(55, 500, 543, 572), fill=(0.88, 0.93, 0.95), color=None)
    pg.insert_text((69, 520), "NET PAY / TAKE-HOME", fontname="hebo", fontsize=10)
    if patch:   # overpaint: bordered patch of a different colour and a different typeface; the value no longer equals gross - deductions
        net = "INR 98,700"
        pg.draw_rect(pymupdf.Rect(327, 508, 499, 539), fill=(0.96, 0.95, 0.91), color=(0.83, 0.81, 0.75), width=0.6)
        pg.insert_text((344, 530), net, fontname="cour", fontsize=15.6)
    else:
        pg.insert_text((337, 530), "INR 89,700", fontname="hebo", fontsize=15.6)
    doc.save(path)
    doc.close()


def main():
    if not os.path.isdir(os.path.join(SAMPLES, "case_clean")):
        subprocess.check_call([sys.executable, os.path.join(ROOT, "synth", "generate.py")])
    for case in ("case_clean", "case_tampered", "case_scan_tampered"):
        dst = os.path.join(DEMO, case)
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst)
        for f in os.listdir(os.path.join(SAMPLES, case)):
            if f.endswith(".jpg") or f == "truth.json":
                shutil.copy2(os.path.join(SAMPLES, case, f), dst)
    for name, patch in (("pdf_payslip_clean", False), ("pdf_payslip_patched", True)):
        dst = os.path.join(DEMO, name)
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst)
        _payslip(os.path.join(dst, "salary_slip.pdf"), patch)
    print("demo_data ready:", sorted(os.listdir(DEMO)))


if __name__ == "__main__":
    main()
