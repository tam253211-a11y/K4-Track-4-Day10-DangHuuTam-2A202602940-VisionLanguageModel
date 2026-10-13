"""Test reward IoU (E5) — viết TRƯỚC khi chạy GRPO. Chạy: python -m pytest -q day10/tests

Đặt ngoài tests/ của đề: giảng viên ghi đè tests/ bằng bản gốc, sửa/thêm vào đó bị −10.
"""
import json

import pytest

from day10.reward import completion_text, compute_reward, iou_reward

POT = {"label": "pothole", "bbox": (100.0, 500.0, 200.0, 600.0)}
CRACK = {"label": "crack", "bbox": (400.0, 700.0, 600.0, 760.0)}


def ans(*boxes):
    return json.dumps([{"label": b["label"], "bbox_2d": list(b["bbox"])} for b in boxes])


def test_perfect_is_one():
    assert compute_reward(ans(POT, CRACK), [POT, CRACK]) == pytest.approx(1.0)


def test_perfect_inside_markdown_fence_is_one():
    assert compute_reward("```json\n" + ans(POT) + "\n```", [POT]) == pytest.approx(1.0)


def test_broken_json_is_zero():
    assert compute_reward('[{"label": "pothole", "bbox_2d": [1, 2', [POT]) == 0.0
    assert compute_reward("I see a pothole.", [POT]) == 0.0


def test_degenerate_box_counts_as_broken():
    assert compute_reward('[{"label":"pothole","bbox_2d":[200,500,100,600]}]', [POT]) == 0.0


def test_wrong_label_gets_only_format_reward():
    wrong = {"label": "manhole", "bbox": POT["bbox"]}
    assert compute_reward(ans(wrong), [POT]) == pytest.approx(0.2)


def test_extra_box_is_penalised_by_denominator():
    # 1 GT, 2 pred (1 khớp hoàn hảo + 1 thừa) → 0.2 + 0.8 · 1/2
    assert compute_reward(ans(POT, CRACK), [POT]) == pytest.approx(0.6)


def test_both_empty_is_one():
    assert compute_reward("[]", []) == pytest.approx(1.0)


def test_empty_prediction_on_damaged_image_is_floor():
    assert compute_reward("[]", [POT, CRACK]) == pytest.approx(0.2)


def test_giant_box_does_not_hack():
    giant = {"label": "pothole", "bbox": (0.0, 0.0, 1000.0, 1000.0)}
    assert compute_reward(ans(giant), [POT]) == pytest.approx(0.2)   # IoU = 0.01 < 0.5


def test_iou_below_threshold_not_matched_and_partial_iou_scaled():
    shifted = {"label": "pothole", "bbox": (150.0, 500.0, 250.0, 600.0)}   # IoU = 1/3
    assert compute_reward(ans(shifted), [POT]) == pytest.approx(0.2)
    close = {"label": "pothole", "bbox": (110.0, 500.0, 210.0, 600.0)}     # IoU = 90/110
    assert compute_reward(ans(close), [POT]) == pytest.approx(0.2 + 0.8 * 90 / 110)


def test_greedy_matching_is_one_to_one():
    # 2 dự đoán trùng nhau cho 1 GT: chỉ 1 cặp được khớp
    assert compute_reward(ans(POT, POT), [POT]) == pytest.approx(0.6)


def test_reward_in_range():
    for c in ["[]", ans(POT), ans(CRACK), "xx", ans(POT, CRACK, POT)]:
        for gt in ([], [POT], [POT, CRACK]):
            r = compute_reward(c, gt)
            assert r == 0.0 or 0.2 <= r <= 1.0


def test_trl_signature_and_conversational_completions():
    gt = json.dumps([{"label": "pothole", "bbox": list(POT["bbox"])}])
    conv = [[{"role": "assistant", "content": ans(POT)}], [{"role": "assistant", "content": "oops"}]]
    assert iou_reward(conv, [gt, gt], trainer_state=None) == pytest.approx([1.0, 0.0])
    assert completion_text([{"role": "assistant", "content": [{"type": "text", "text": "[]"}]}]) == "[]"
