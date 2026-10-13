"""Reward IoU kiểm chứng được cho GRPO (GUIDE bước 9). Python thuần, test không cần GPU."""
import json

from scripts.score import iou_xyxy, parse_output

IOU_THR = 0.5


def compute_reward(completion, gt_boxes):
    """gt_boxes: [{"label", "bbox": (x1,y1,x2,y2)}] thang 0–1000 (xem common.gt_boxes_norm1000).

    0.0 nếu JSON hỏng; 1.0 nếu cả GT và dự đoán đều rỗng;
    còn lại 0.2 + 0.8 · ΣIoU(cặp khớp cùng lớp, IoU ≥ 0.5, ghép tham lam) / max(n_gt, n_pred, 1).
    Mẫu số max(...) phạt box bịa; ngưỡng 0.5 làm box khổng lồ không ăn điểm.
    """
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    total_iou = 0.0
    for lab in {g["label"] for g in gt_boxes} | {p["label"] for p in pred}:
        g = [x["bbox"] for x in gt_boxes if x["label"] == lab]
        p = [x["bbox"] for x in pred if x["label"] == lab]
        pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(g) for j, b in enumerate(p)),
                       reverse=True)
        used_g, used_p = set(), set()
        for v, i, j in pairs:
            if v < IOU_THR:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            total_iou += v
    return 0.2 + 0.8 * total_iou / max(len(gt_boxes), len(pred), 1)


def completion_text(c):
    """TRL truyền completion dạng chuỗi (standard) hoặc list message (conversational)."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        content = c[-1].get("content", "")
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content or ""
    return str(c)


def iou_reward(completions, gt, **kwargs):
    """Chữ ký reward function của trl.GRPOTrainer: cột 'gt' của dataset (JSON string) tới qua kwargs."""
    out = []
    for c, g in zip(completions, gt):
        boxes = [{"label": b["label"], "bbox": tuple(b["bbox"])} for b in json.loads(g)]
        out.append(compute_reward(completion_text(c), boxes))
    return out
