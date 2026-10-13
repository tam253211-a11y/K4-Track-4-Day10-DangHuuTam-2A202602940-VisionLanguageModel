"""Câu 3 · Kiểm chứng lệch thang toạ độ bằng chính trọng tài score.evaluate (không ghi report/evidence).

1) Nhãn đúng (oracle) chấm đúng thang → F1 = 1; chấm sai thang → F1 ≈ 0 dù JSON hợp lệ 100%.
2) Dự đoán thật của model (thang 0–1000) chấm như pixel → F1 sụp.
    python -m day10.coord_check --pred work/e1.jsonl
"""
import argparse
import json

from day10.common import load_split, objects_to_answer
from scripts.score import evaluate


def _row(name, res):
    m = res["micro"]
    print(f"  {name:48s} valid={res['parse_valid_rate']:.3f} F1={m['f1']:.3f} "
          f"(TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred")
    ap.add_argument("--split", default="test")
    a = ap.parse_args(argv)
    entries = load_split(a.split)
    norm = {e["image_id"]: objects_to_answer(e["objects"]) for e in entries}
    pix = {e["image_id"]: objects_to_answer(e["objects"], "pixel", e["width"], e["height"]) for e in entries}
    print(f"Oracle (nhãn đúng làm dự đoán) trên {a.split}:")
    _row("nhãn 0–1000, chấm --coord norm1000", evaluate(entries, norm, "norm1000", n_boot=1))
    _row("nhãn pixel,  chấm --coord pixel", evaluate(entries, pix, "pixel", n_boot=1))
    _row("nhãn pixel,  chấm --coord norm1000 (SAI THANG)", evaluate(entries, pix, "norm1000", n_boot=1))
    _row("nhãn 0–1000, chấm --coord pixel    (SAI THANG)", evaluate(entries, norm, "pixel", n_boot=1))
    if a.pred:
        with open(a.pred, encoding="utf-8") as f:
            preds = {r["image_id"]: r["raw"] for r in map(json.loads, filter(str.strip, f))}
        print(f"Dự đoán thật {a.pred}:")
        _row("chấm --coord norm1000 (đúng)", evaluate(entries, preds, "norm1000", n_boot=1))
        _row("chấm --coord pixel    (SAI THANG)", evaluate(entries, preds, "pixel", n_boot=1))


if __name__ == "__main__":
    main()
