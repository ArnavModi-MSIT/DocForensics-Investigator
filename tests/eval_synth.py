"""Score the pipeline on synthetic ground truth, per capture profile.
Metrics: false positives on clean cases, recall of injected issues (visual vs content), field-extraction
accuracy (exact match after normalisation), document-classification accuracy.
Run: python tests/eval_synth.py [case ...]      (default: every case in samples/ that has truth.json)
Tip: DOCFOR_EMBED=off|auto  to compare without / with the embedding model."""
import glob, json, os, re, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from docforensics.pipeline import run_case
from docforensics.embed_classifier import status

ROOT = os.path.join(os.path.dirname(__file__), "..")
KEY = {"Net Pay": "net_pay", "Date of Birth": "dob", "Account Holder": "name", "ID Number": "id_number",
       "Declared Monthly Income": "declared_income", "Total Deductions": "deductions",
       "Salary Credit": "salary_credit", "Closing Balance": "closing_balance"}


def norm(s):
    return re.sub(r"[\s,]|rs\.?", "", str(s).lower())


def eval_case(case):
    src = os.path.join(ROOT, "samples", case)
    r = run_case(sorted(glob.glob(os.path.join(src, "*.jpg"))), os.path.join(ROOT, "output", case))
    truth = json.load(open(os.path.join(src, "truth.json")))
    ft = json.load(open(os.path.join(src, "fields_truth.json")))
    det = []
    for t in truth:
        key = KEY[t["label"]]
        kinds = ("visual",) if t["kind"] == "visual_patch" else ("cross", "intra")      # the RIGHT kind of finding must exist
        ok = any(any(x["doc_id"] == t["doc"] and x["field"] == key for x in f.refs) and f.kind in kinds
                 for f in r["findings"])
        det.append({**t, "detected": ok})
    fe_ok = fe_n = 0
    miss = []
    for d in r["docs"]:
        for key, val in ft.get(d.doc_id, {}).items():
            fe_n += 1
            f = d.get(key)
            if f and norm(f.raw) == norm(val):
                fe_ok += 1
            else:
                miss.append(f"{d.doc_id}.{key}: want {val!r} got {f.raw if f else None!r}")
    cls_ok = sum(d.doc_type == d.doc_id for d in r["docs"])
    return {"case": case, "findings": [f"{f.fid} {f.title}" for f in r["findings"]], "n_findings": len(r["findings"]),
            "truth": len(truth), "detected": sum(x["detected"] for x in det), "detail": det,
            "extract_ok": fe_ok, "extract_n": fe_n, "extract_miss": miss, "cls_ok": cls_ok, "cls_n": len(r["docs"])}


def kind_of(case):
    if "skilled" in case:
        return "skilled"
    if "clean" in case:
        return "clean"
    return "forged"          # tampered / naive


def check(res, min_extract=0.90):
    """Regression gate. clean: no findings at all; forged: every injected issue found with the right finding kind;
    all: classification perfect, extraction >= 90%. Skilled forgeries are a KNOWN limitation (not required to be
    detected) but must not produce a pile of unrelated findings."""
    fails = []
    for r in res:
        k = kind_of(r["case"])
        if r["cls_ok"] != r["cls_n"]:
            fails.append(f"{r['case']}: classification {r['cls_ok']}/{r['cls_n']}")
        if r["extract_n"] and r["extract_ok"] / r["extract_n"] < min_extract:
            fails.append(f"{r['case']}: extraction {r['extract_ok']}/{r['extract_n']} below {min_extract:.0%}")
        if k == "clean" and r["n_findings"] != 0:
            fails.append(f"{r['case']}: {r['n_findings']} false positive(s) {r['findings']}")
        if k == "forged" and r["detected"] != r["truth"]:
            fails.append(f"{r['case']}: detected {r['detected']}/{r['truth']} injected issues")
        if k == "skilled" and r["n_findings"] > 3:
            fails.append(f"{r['case']}: {r['n_findings']} unrelated findings on a consistent forgery")
    return fails


def main():
    cases = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "samples", "*", "truth.json")))
    print("embedding:", status()["backend"] or "OFF (keywords + structure only)")
    res, t0 = [], time.time()
    for c in cases:
        res.append(eval_case(c)); print(f"  done {c} ({time.time() - t0:.0f}s)", flush=True)
    print(f"\n{'case':22} {'classif':>8} {'extract':>9} {'issues found':>13} {'findings':>9}")
    for r in res:
        iss = f"{r['detected']}/{r['truth']}" if r["truth"] else "-"
        fp = f"{r['n_findings']} (FP)" if not r["truth"] else str(r["n_findings"])
        print(f"{r['case']:22} {r['cls_ok']}/{r['cls_n']:>5} {r['extract_ok']}/{r['extract_n']:>6} {iss:>13} {fp:>9}")
    groups = {}
    for r in res:
        m = re.match(r"case_hard_(digital|scan_hard|scan|rough)_(clean|naive|skilled)_", r["case"])
        if m:
            g = groups.setdefault((m.group(1), m.group(2)), dict(cases=0, fp=0, truth=0, det=0, ex_ok=0, ex_n=0, cls_ok=0, cls_n=0))
            g["cases"] += 1; g["ex_ok"] += r["extract_ok"]; g["ex_n"] += r["extract_n"]; g["cls_ok"] += r["cls_ok"]; g["cls_n"] += r["cls_n"]
            if r["truth"]:
                g["truth"] += r["truth"]; g["det"] += r["detected"]
            else:
                g["fp"] += r["n_findings"]
    if groups:
        print(f"\nHELD-OUT SUMMARY  {'capture':10} {'forgery':8} {'cases':>5} {'issues found':>13} {'false pos':>10} {'extract':>9} {'classif':>8}")
        for (prof, kind), g in sorted(groups.items()):
            iss = f"{g['det']}/{g['truth']}" if g["truth"] else "-"
            fp = g["fp"] if kind == "clean" else "-"
            print(f"                  {prof:10} {kind:8} {g['cases']:>5} {iss:>13} {fp!s:>10} {g['ex_ok']}/{g['ex_n']:<4} {g['cls_ok']}/{g['cls_n']}")
    for r in res:
        bad = [d for d in r["detail"] if not d["detected"]]
        if bad or (not r["truth"] and r["n_findings"]) or r["extract_miss"]:
            print(f"\n[{r['case']}]")
            for d in bad:
                print(f"   MISSED  {d['kind']:16} {d['doc']}.{d['label']}")
            if not r["truth"]:
                for f in r["findings"]:
                    print(f"   FALSE POSITIVE  {f}")
            for m in r["extract_miss"][:8]:
                print(f"   extraction: {m}")
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "output", "eval.json"), "w"), indent=2)
    failures = check(res)
    print("\nEXPECTATIONS:", "ALL PASSED" if not failures else f"{len(failures)} FAILED")
    for f in failures:
        print("   FAIL:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
