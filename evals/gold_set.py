"""
Judge validation against a hand-labelled gold set.

An LLM judge is only trustworthy if it agrees with a human on cases where the
answer is known. This runs the judge over evals/gold_labels.json (claim + source
+ human `gold` verdict), then reports:
  - overall accuracy
  - per-class precision/recall
  - Cohen's kappa (chance-corrected agreement) over {SUPPORTED, UNSUPPORTED, CONTRADICTED}
  - the confusion matrix and every disagreement

Rule of thumb: kappa >= 0.6 is substantial agreement; below that, retune the judge
prompt before trusting the step results.
"""

from __future__ import annotations

import json
from pathlib import Path

from .judge import Judge

GOLD = Path(__file__).resolve().parent / "gold_labels.json"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LABELS = ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED")


def cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if n == 0:
        return 0.0
    po = sum(1 for g, j in pairs if g == j) / n
    gold_freq = {l: sum(1 for g, _ in pairs if g == l) / n for l in LABELS}
    judge_freq = {l: sum(1 for _, j in pairs if j == l) / n for l in LABELS}
    pe = sum(gold_freq[l] * judge_freq[l] for l in LABELS)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def run(judge: Judge | None = None) -> dict:
    judge = judge or Judge()
    items = json.loads(GOLD.read_text())

    pairs: list[tuple[str, str]] = []
    rows = []
    for it in items:
        j = judge.grade(it["claim"], it["source"])
        pairs.append((it["gold"], j.verdict))
        rows.append({
            "id": it["id"], "gold": it["gold"], "judge": j.verdict,
            "agree": it["gold"] == j.verdict, "claim": it["claim"][:70],
        })

    n = len(pairs)
    accuracy = sum(1 for g, jd in pairs if g == jd) / n if n else 0.0
    kappa = cohens_kappa(pairs)

    # confusion matrix gold(row) x judge(col)
    confusion = {g: {jl: 0 for jl in LABELS} for g in LABELS}
    for g, jd in pairs:
        confusion[g][jd] += 1

    # per-class precision/recall
    per_class = {}
    for l in LABELS:
        tp = sum(1 for g, jd in pairs if g == l and jd == l)
        judged = sum(1 for _, jd in pairs if jd == l)
        actual = sum(1 for g, _ in pairs if g == l)
        per_class[l] = {
            "precision": round(tp / judged, 3) if judged else None,
            "recall": round(tp / actual, 3) if actual else None,
            "support": actual,
        }

    return {
        "summary": {
            "step": "judge_gold_validation",
            "n": n,
            "accuracy": round(accuracy, 3),
            "cohens_kappa": round(kappa, 3),
            "interpretation": _kappa_label(kappa),
            "per_class": per_class,
            "confusion_gold_x_judge": confusion,
        },
        "disagreements": [r for r in rows if not r["agree"]],
        "rows": rows,
        "usage": judge.cost_report(),
    }


def _kappa_label(k: float) -> str:
    if k >= 0.8:
        return "almost perfect"
    if k >= 0.6:
        return "substantial"
    if k >= 0.4:
        return "moderate"
    return "poor — retune the judge before trusting results"


def _print(res: dict) -> None:
    s = res["summary"]
    print("=" * 68)
    print("JUDGE GOLD-SET VALIDATION")
    print("=" * 68)
    print(f"  examples          : {s['n']}")
    print(f"  accuracy          : {s['accuracy']*100:.1f}%")
    print(f"  Cohen's kappa     : {s['cohens_kappa']}  ({s['interpretation']})")
    print("  per-class (precision / recall / support):")
    for l, m in s["per_class"].items():
        print(f"    {l:13s}: P={m['precision']} R={m['recall']} n={m['support']}")
    if res["disagreements"]:
        print(f"\n  disagreements ({len(res['disagreements'])}):")
        for d in res["disagreements"]:
            print(f"    [{d['id']}] gold={d['gold']} judge={d['judge']} :: {d['claim']}")
    else:
        print("\n  no disagreements — judge matches every gold label")


if __name__ == "__main__":
    res = run()
    _print(res)
    u = res["usage"]
    print(f"\n  cost: ${u['cost_usd']['total']} ({u['judge_calls']} calls)")
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "gold_validation.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"  wrote {out}")
