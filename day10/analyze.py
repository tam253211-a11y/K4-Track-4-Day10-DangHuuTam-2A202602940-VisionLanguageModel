"""Phân tích cho REPORT.md: bảng tổng hợp, ảnh minh hoạ đúng/sai, lỗi theo lớp, reward hacking.

    python -m day10.analyze table                              # bảng markdown từ report/evidence/*.json
    python -m day10.analyze figures --pred work/e1.jsonl --tag e1
    python -m day10.analyze classes --pred work/e1.jsonl       # Câu 9: F1/recall theo lớp × kích thước box
    python -m day10.analyze hacking --pred work/e1.jsonl work/e5.jsonl   # E5: [] / box khổng lồ / số box
    python -m day10.analyze e0 --pred work/e0.jsonl            # Câu 2: phân loại lỗi của zero-shot
"""
import argparse
import json
import statistics
from pathlib import Path

from day10.common import CLASSES, DATA, ROOT, load_split, read_json
from scripts.score import IOU_THR, _gt_boxes, _to_norm1000, iou_xyxy, parse_output

SETUP_DESC = {"E0": "zero-shot", "E1": "LoRA LLM", "E2": "+ projector", "E3": "+ LoRA ViT", "E5": "GRPO"}


def load_preds(path):
    with open(path, encoding="utf-8") as f:
        return {r["image_id"]: r["raw"] for r in map(json.loads, filter(str.strip, f))}


def _f(x, nd=3):
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------- bảng tổng hợp
def table(ev_dir=ROOT / "report" / "evidence"):
    order = lambda p: (p.stem[:2], p.stem)  # noqa: E731
    lines = ["| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole "
             "| F1 crack | F1 manhole | parse_valid |", "|" + "---|" * 12]
    for p in sorted(Path(ev_dir).glob("e*.json"), key=order):
        d = read_json(p)
        m, pc, meta = d["micro"], d["per_class"], d.get("meta", {})
        name = d["name"]
        desc = SETUP_DESC.get(name, name)
        lines.append(
            f"| {name} | {desc} | {meta.get('trainable_params', 0):,} | {meta.get('peak_vram_gb', '')} "
            f"| {meta.get('train_minutes', '')} | {_f(m['f1'])} | {_f(d['f1_ci95'][0])}–{_f(d['f1_ci95'][1])} "
            f"| {_f(d['fp_per_image'], 2)} | {_f(pc['pothole']['f1'])} | {_f(pc['crack']['f1'])} "
            f"| {_f(pc['manhole']['f1'])} | {_f(d['parse_valid_rate'])} |")
    print("\n".join(lines))
    return lines


# ---------------------------------------------------------------- ghép cặp chi tiết
def match_detail(gt, pred):
    """Như scripts.score._match nhưng trả về cờ TP cho từng box (để vẽ)."""
    gt_ok, pr_ok, ious = [False] * len(gt), [False] * len(pred), {}
    for lab in {g["label"] for g in gt} | {p["label"] for p in pred}:
        gi = [i for i, g in enumerate(gt) if g["label"] == lab]
        pj = [j for j, p in enumerate(pred) if p["label"] == lab]
        pairs = sorted(((iou_xyxy(gt[i]["bbox"], pred[j]["bbox"]), i, j) for i in gi for j in pj), reverse=True)
        for v, i, j in pairs:
            if v < IOU_THR:
                break
            if gt_ok[i] or pr_ok[j]:
                continue
            gt_ok[i] = pr_ok[j] = True
            ious[j] = v
    return gt_ok, pr_ok, ious


def per_image(pred_path, split="test", coord=None):
    preds = load_preds(pred_path)
    info = Path(pred_path).with_suffix(".info.json")
    coord = coord or (read_json(info).get("coord") if info.exists() else "norm1000")
    rows = []
    for e in load_split(split):
        raw = preds.get(e["image_id"])
        parsed = parse_output(raw) if raw is not None else None
        pred = _to_norm1000(parsed, coord, e["width"], e["height"]) if parsed else []
        gt = _gt_boxes(e)
        gt_ok, pr_ok, ious = match_detail(gt, pred)
        rows.append({"entry": e, "raw": raw, "valid": parsed is not None, "gt": gt, "pred": pred,
                     "gt_ok": gt_ok, "pr_ok": pr_ok, "ious": ious,
                     "tp": sum(pr_ok), "fp": len(pred) - sum(pr_ok), "fn": len(gt) - sum(gt_ok)})
    return rows


# ---------------------------------------------------------------- ảnh minh hoạ
def draw(row, title=None):
    from PIL import Image, ImageDraw
    e = row["entry"]
    img = Image.open(DATA / e["image_path"]).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)
    px = lambda b: [b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000]  # noqa: E731
    for g, ok in zip(row["gt"], row["gt_ok"]):          # GT: xanh lá nếu khớp, vàng nếu bị bỏ sót (FN)
        d.rectangle(px(g["bbox"]), outline=(0, 220, 0) if ok else (255, 215, 0), width=4)
    for j, (p, ok) in enumerate(zip(row["pred"], row["pr_ok"])):   # dự đoán: xanh dương TP, đỏ FP
        b = px(p["bbox"])
        col = (30, 144, 255) if ok else (255, 40, 40)
        d.rectangle(b, outline=col, width=2)
        d.text((b[0] + 2, max(0, b[1] - 11)), f"{p['label']}" + (f" {row['ious'][j]:.2f}" if ok else ""), fill=col)
    return img


def figures(pred_path, tag, split="test", n_good=3, n_bad=3, out_dir=ROOT / "report" / "figures"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = per_image(pred_path, split)
    good = sorted([r for r in rows if r["gt"] and r["fp"] == 0 and r["fn"] == 0],
                  key=lambda r: (-len(r["gt"]), r["entry"]["image_id"]))[:n_good]
    fp_heavy = sorted([r for r in rows if r["fp"] > 0], key=lambda r: (-r["fp"], r["entry"]["image_id"]))
    fn_heavy = sorted([r for r in rows if r["fn"] > 0], key=lambda r: (-r["fn"], r["entry"]["image_id"]))
    bad, seen = [], set()
    for r in [x for pair in zip(fp_heavy, fn_heavy) for x in pair] + fp_heavy + fn_heavy:
        if r["entry"]["image_id"] not in seen and len(bad) < n_bad:
            bad.append(r)
            seen.add(r["entry"]["image_id"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    picked = [("dung", r) for r in good] + [("sai", r) for r in bad]
    fig, axes = plt.subplots(2, 3, figsize=(19, 7.5))
    captions = []
    for k, (ax, (kind, r)) in enumerate(zip(axes.flat, picked)):
        img = draw(r)
        fn = out_dir / f"{tag}_{kind}_{k + 1}.png"
        img.save(fn)
        cap = (f"{tag.upper()} · {r['entry']['image_id']} · TP={r['tp']} FP={r['fp']} FN={r['fn']}"
               + ("" if r["valid"] else " · JSON hỏng"))
        captions.append((fn.relative_to(ROOT).as_posix(), cap))
        ax.imshow(img)
        ax.set_title(cap, fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[len(picked):]:
        ax.axis("off")
    fig.suptitle("GT khớp = xanh lá · GT bỏ sót (FN) = vàng · dự đoán đúng (TP, kèm IoU) = xanh dương · "
                 "dự đoán sai (FP) = đỏ", fontsize=10)
    plt.tight_layout()
    grid = out_dir / f"{tag}_examples.png"
    plt.savefig(grid, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] {grid}")
    for fn, cap in captions:
        print(f"![{cap}]({fn})")
    return captions


# ---------------------------------------------------------------- Câu 9: theo lớp × kích thước
SIZE_BINS = [(0, 0.3, "<0.3%"), (0.3, 1.0, "0.3–1%"), (1.0, 100, "≥1%")]


def area_pct(b):
    return (b[2] - b[0]) * (b[3] - b[1]) / 1e4   # thang 0–1000 → % diện tích khung hình


def classes(pred_path=None, split="test"):
    print("Thống kê nhãn (diện tích box tính theo % khung hình):")
    for s in ("train", "val", "test"):
        for c in CLASSES:
            areas = [o["yolo"][2] * o["yolo"][3] * 100 for r in load_split(s) for o in r["objects"] if o["label"] == c]
            if areas:
                print(f"  {s:5s} {c:8s} n={len(areas):5d}  median={statistics.median(areas):.2f}%  "
                      f"p25={statistics.quantiles(areas, n=4)[0]:.2f}%  p75={statistics.quantiles(areas, n=4)[2]:.2f}%")
    if not pred_path:
        return
    rows = per_image(pred_path, split)
    print(f"\nRecall theo lớp × kích thước ({pred_path}, IoU≥{IOU_THR}):")
    for c in CLASSES:
        parts = []
        for lo, hi, name in SIZE_BINS:
            hits = [ok for r in rows for g, ok in zip(r["gt"], r["gt_ok"])
                    if g["label"] == c and lo <= area_pct(g["bbox"]) < hi]
            parts.append(f"{name}: {sum(hits)}/{len(hits)}" + (f"={sum(hits) / len(hits):.2f}" if hits else ""))
        print(f"  {c:8s} " + " · ".join(parts))
    conf = {}
    for r in rows:      # FP có trùng vị trí GT lớp khác không (nhầm lớp) hay bịa hẳn
        for p, ok in zip(r["pred"], r["pr_ok"]):
            if ok:
                continue
            best = max(((iou_xyxy(p["bbox"], g["bbox"]), g["label"]) for g in r["gt"]), default=(0, None))
            key = f"{p['label']}→nhầm lớp {best[1]}" if best[0] >= IOU_THR else (
                f"{p['label']}→lệch vị trí (0<IoU<{IOU_THR})" if best[0] > 0 else f"{p['label']}→bịa (IoU=0)")
            conf[key] = conf.get(key, 0) + 1
    print("\nPhân loại FP:")
    for k, v in sorted(conf.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")


# ---------------------------------------------------------------- Câu 2: lỗi E0
def e0_errors(pred_path, split="test"):
    import re
    preds = load_preds(pred_path)
    n = len(preds)
    fence = sum(1 for r in preds.values() if "```" in r)
    extra_text = sum(1 for r in preds.values() if r.strip() and not r.strip().startswith(("[", "```")))
    broken = {iid: r for iid, r in preds.items() if parse_output(r) is None}
    rows = per_image(pred_path, split)
    labels, maxc, n_boxes = {}, [], 0
    for r in rows:
        for p in r["pred"]:
            labels[p["label"]] = labels.get(p["label"], 0) + 1
            maxc.append(max(p["bbox"]))
            n_boxes += 1
    print(f"n={n} · JSON hỏng={len(broken)} ({len(broken) / max(n, 1):.1%}) · bọc ```={fence} · có chữ thừa={extra_text}")
    print(f"tổng box dự đoán={n_boxes} · GT={sum(len(r['gt']) for r in rows)} · nhãn dự đoán={labels}")
    if maxc:
        print(f"toạ độ lớn nhất: median={statistics.median(maxc):.0f} (thang 0–1000; >1000 = sai thang)")
        print(f"box có toạ độ >1000: {sum(1 for v in maxc if v > 1000)}")
    for iid, r in list(broken.items())[:3]:
        print(f"--- JSON hỏng {iid}: {r[:300]!r}")
    keys = re.findall(r'"(\w+)"\s*:', " ".join(list(broken.values())[:50]))
    if keys:
        print("khoá xuất hiện trong output hỏng:", sorted(set(keys))[:15])


# ---------------------------------------------------------------- E5: reward hacking
def hacking(pred_paths, split="test"):
    print("| file | parse_valid | tỷ lệ '[]' | ảnh GT rỗng | box/ảnh (pred) | box/ảnh (GT) | diện tích box TB | box >25% ảnh |")
    print("|---|---|---|---|---|---|---|---|")
    for p in pred_paths:
        rows = per_image(p, split)
        n = len(rows)
        empty = sum(1 for r in rows if r["valid"] and not r["pred"])
        areas = [area_pct(b["bbox"]) for r in rows for b in r["pred"]]
        print(f"| {p} | {sum(r['valid'] for r in rows) / n:.3f} | {empty / n:.3f} "
              f"| {sum(1 for r in rows if not r['gt']) / n:.3f} | {sum(len(r['pred']) for r in rows) / n:.2f} "
              f"| {sum(len(r['gt']) for r in rows) / n:.2f} | {statistics.mean(areas) if areas else 0:.2f}% "
              f"| {sum(1 for a in areas if a > 25)} |")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table")
    f = sub.add_parser("figures")
    f.add_argument("--pred", required=True)
    f.add_argument("--tag", required=True)
    c = sub.add_parser("classes")
    c.add_argument("--pred")
    z = sub.add_parser("e0")
    z.add_argument("--pred", required=True)
    h = sub.add_parser("hacking")
    h.add_argument("--pred", nargs="+", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "table":
        table()
    elif a.cmd == "figures":
        figures(a.pred, a.tag)
    elif a.cmd == "classes":
        classes(a.pred)
    elif a.cmd == "e0":
        e0_errors(a.pred)
    else:
        hacking(a.pred)


if __name__ == "__main__":
    main()
