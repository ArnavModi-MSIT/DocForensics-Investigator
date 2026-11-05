"""Score the system against a ground-truth pack (synthetic documents with planted findings).

    python -m tests.eval_pack [PACK_DIR] [--cases A,B,C,D] [--keep]

PACK_DIR holds GROUND_TRUTH.json and case_A/ ... case_D/ folders (default: output/_inputs/pack, git-ignored: it is not part of the
repository and the answer key must never be placed in a folder that is uploaded to the system).

Reported separately, as the pack's own guide recommends:
  extraction    for every source field box in the answer key: was a field found there, and does its text equal the text of the PDF in that box?
  controls      findings that are not observations on the clean cases (A, D): false positives
  planted       per planted finding: found (right kind, evidence boxes touched in every document) or missed, with localisation overlap
  precision     substantive findings (severity medium/high or status other than 'observed') that match no planted finding
  language      titles that assert fraud / forgery instead of an inconsistency or an observation
"""
import glob, json, os, re, shutil, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from docforensics.pipeline import run_case, EXTS            # noqa: E402
from docforensics.pdf_layer import SCALE                    # noqa: E402

FIELD_KEYS = {   # answer-key field -> our field keys (first found wins)
    "name": ["name"], "birth_date": ["dob"], "personal_id": ["id_number"], "employee_id": ["employee_id"],
    "residential_address": ["address"], "employer": ["employer"], "monthly_gross": ["declared_income", "monthly_gross"],
    "annual_gross": ["annual_gross"], "account": ["account_no"], "loan_requested": ["credit_amount"], "loan_term": ["term"],
    "app_id": ["case_ref"], "issued": ["issue_date"], "review_due": ["review_date"], "employment_type": ["employment_type"],
    "start_date": ["start_date"], "pay_period": ["pay_month"], "pay_date": ["pay_date"], "net_pay": ["net_pay"],
    "statement_date": ["period"], "class": ["account_type"], "salary_credit": ["salary_credit"],
}
KIND_OF = {"contextual_arithmetic": lambda f: f.kind == "intra" and "arithmetic" in (f.method or "") + f.title.lower(),
           "cross_document_conflict": lambda f: f.kind == "cross",
           "visual_forensic_observation": lambda f: f.kind == "visual"}
FRAUD_WORDS = re.compile(r"(?i)\b(fraud\w*|forg(ed|ery)|fake|fabricated|counterfeit)\b")


def norm(t):
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def overlap(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return 0.0 if w <= 0 or h <= 0 else w * h / max(1e-9, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def pdf_text(path, box):
    import pymupdf
    pg = pymupdf.open(path)[0]
    return " ".join(pg.get_textbox(pymupdf.Rect(*box)).split())


def run(pack, cases, keep):
    gt = json.load(open(pack / "GROUND_TRUTH.json"))
    planted = gt["expected_findings"]
    summary = {"extraction": [0, 0, 0], "fp_controls": 0, "planted_found": 0, "planted_total": 0, "fp_substantive": 0, "fraud_titles": 0}
    for case in cases:
        folder = pack / f"case_{case}"
        if not folder.is_dir():
            continue
        paths = sorted(p for p in glob.glob(str(folder / "*")) if p.lower().endswith(EXTS))
        if case == "D":                                              # the PDF and the JPG are the same page: submit one
            paths = [p for p in paths if p.endswith(".pdf")] or paths[:1]
        out = tempfile.mkdtemp(prefix=f"pack_{case}_") if not keep else str(ROOT / "output" / f"pack_{case}")
        r = run_case(paths, out, case_id=f"case_{case}")
        docs, finds = {d.doc_id: d for d in r["docs"]}, r["findings"]
        print(f"\n=== case {case}: {len(docs)} page(s), {len(finds)} finding(s)")
        # ---- extraction vs the PDF's own text in the answer-key boxes
        for fname, fields in gt["source_field_boxes"].get(f"case_{case}", {}).items():
            stem = Path(fname).stem
            d = docs.get(stem)
            if d is None:
                continue
            for gname, box in fields.items():
                keys = FIELD_KEYS.get(gname)
                if not keys:
                    continue
                want = pdf_text(str(folder / fname), box)
                got = next((f for k in keys for f in d.fields if f.key == k), None)
                located = got is not None and overlap([v / SCALE for v in got.bbox], box) >= 0.3
                same = got is not None and norm(got.raw) and (norm(got.raw) in norm(want) or norm(want) in norm(got.raw))
                summary["extraction"][0] += 1
                summary["extraction"][1] += int(located)
                summary["extraction"][2] += int(located and same)
                if not (located and same):
                    print(f"   extraction miss  {stem}.{gname}: want {want!r} got {got.raw if got else None!r} ({'located' if located else 'not located'})")
        # ---- planted findings
        mine = [p for p in planted if p["case"] == f"case_{case}"]
        matched = set()
        for p in mine:
            summary["planted_total"] += 1
            need = {Path(e["file"]).stem for e in p["evidence"]}
            best = None
            for f in finds:
                if f.kind == "missing" or not KIND_OF[p["kind"]](f):
                    continue
                touched, ious = set(), []
                for e in p["evidence"]:
                    for rf in f.refs:
                        if rf.get("bbox") and rf["doc_id"] == Path(e["file"]).stem and overlap([v / SCALE for v in rf["bbox"]], e["box"]) >= 0.3:
                            touched.add(rf["doc_id"]); ious.append(iou([v / SCALE for v in rf["bbox"]], e["box"]))
                if touched == need and (best is None or max(ious) > best[1]):
                    best = (f, max(ious))
            if best:
                matched.add(best[0].fid)
                summary["planted_found"] += 1
                f = best[0]
                print(f"   FOUND  {p['id']}: {f.fid} [{f.severity}|{f.confidence:.2f}|{f.status}] {f.title}  (best box IoU {best[1]:.2f})")
            else:
                print(f"   MISSED {p['id']}: {p['description']}")
        # ---- everything else
        for f in sorted(finds, key=lambda x: x.fid):
            if f.fid in matched:
                continue
            substantive = f.severity in ("medium", "high") or f.status != "observed"
            if FRAUD_WORDS.search(f.title):
                summary["fraud_titles"] += 1
            tag = "  extra  "
            if substantive and f.kind != "missing" or (substantive and f.severity == "high"):
                summary["fp_substantive"] += 1
                tag = "  FALSE+ "
                if case in ("A", "D"):
                    summary["fp_controls"] += 1
            print(f"{tag}{f.fid} [{f.severity}|{f.confidence:.2f}|{f.status}] {f.title}")
        if not keep:
            shutil.rmtree(out, ignore_errors=True)
    e = summary["extraction"]
    print("\n=== summary")
    print(f"extraction: {e[2]}/{e[0]} fields located with the right text ({e[1]}/{e[0]} located)")
    print(f"planted findings: {summary['planted_found']}/{summary['planted_total']} found")
    print(f"substantive findings not in the answer key: {summary['fp_substantive']} (of which on clean controls: {summary['fp_controls']})")
    print(f"titles asserting fraud/forgery: {summary['fraud_titles']}")
    return summary


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cases = "ABCD"
    for a in sys.argv[1:]:
        if a.startswith("--cases"):
            cases = a.split("=", 1)[1].replace(",", "")
    pack = Path(args[0]) if args else ROOT / "output" / "_inputs" / "pack"
    if not (pack / "GROUND_TRUTH.json").exists():
        print(f"No GROUND_TRUTH.json in {pack}; nothing to score.")
        return 0
    s = run(pack, cases, "--keep" in sys.argv)
    return 0 if s["fp_controls"] == 0 and s["planted_found"] == s["planted_total"] else 1


if __name__ == "__main__":
    sys.exit(main())
