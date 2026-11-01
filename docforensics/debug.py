"""Read-only explanation of what the pipeline 'saw' for a page (does not change any result)."""
import re
from .extract import DOC_TYPES, EXPECTED, MIN_FIELDS, _best_label, keyword_hits
from .ocr import group_lines
from .embed_classifier import status


def explain(doc):
    low = " ".join(w.text.lower() for w in doc.words)
    info = doc.class_info or {}
    sem = (info.get("semantic") or {}).get("sims") or {}
    found = {f.key for f in doc.fields}
    types = []
    for t, kws in DOC_TYPES.items():
        hit = keyword_hits(low, kws)
        types.append({"type": t, "keyword_score": round(len(hit) / len(kws), 2), "keywords_found": ", ".join(hit),
                      "keywords_missing": ", ".join(k for k in kws if k not in hit),
                      "semantic_sim": sem.get(t), f"expected_fields_found (need {MIN_FIELDS})":
                      f"{len(found & EXPECTED[t])}/{len(EXPECTED[t])}: " + ", ".join(sorted(found & EXPECTED[t]))})
    others = [{"type": t, "semantic_sim": v} for t, v in sorted(sem.items(), key=lambda kv: -kv[1]) if t.startswith("other:")]
    lines = []
    for ln in group_lines(doc.words):
        cy = sum((w.bbox[1] + w.bbox[3]) / 2 for w in ln) / len(ln)
        hh = (ln[0].bbox[3] - ln[0].bbox[1]) * 0.6
        hit = next((f for f in doc.fields if abs((f.bbox[1] + f.bbox[3]) / 2 - cy) < hh), None)
        near = ""
        if not hit:                                    # show the nearest dictionary label for lines that did not match
            for n in range(1, min(4, len(ln) - 1) + 1):
                key, sc = _best_label(" ".join(w.text for w in ln[:n]))
                if key:
                    near = f"{key} ({sc:.0f})"
        lines.append({"text": re.sub(r"\b\d{4} \d{4} (\d{4})\b", r"XXXX XXXX \1", " ".join(w.text for w in ln)),
                      "ocr_conf": round(sum(w.conf for w in ln) / len(ln), 2),
                      "field": hit.key if hit else "", "matched_by": hit.match if hit else "",
                      "label_score": (hit.label_sim if hit else ""), "note": near})
    return {"predicted": doc.doc_type, "best_guess": doc.best_guess, "confidence": doc.type_conf,
            "recognised": doc.doc_type != "unknown", "decision": info.get("decision", ""),
            "types": types, "other_classes": others, "lines": lines,
            "n_words": len(doc.words), "n_fields": len(doc.fields), "embedding": status()}
