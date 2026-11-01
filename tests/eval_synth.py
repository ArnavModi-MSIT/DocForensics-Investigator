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
KEY = {"Net Pay": "net_pay", "Date of Birth": "dob", "Account Holder": "name",
       "ID Number": "id_number", "Declared Monthly Income": "declared_income"}


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
        want = "visual" if t["kind"] == "visual_patch" else None
        ok = any(any(x["doc_id"] == t["doc"] and x["field"] == key for x in f.refs) and (want is None or f.kind == want)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
