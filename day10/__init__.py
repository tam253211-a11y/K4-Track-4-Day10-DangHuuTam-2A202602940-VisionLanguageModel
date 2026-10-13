"""Mã của học viên cho Lab Day 10 (Qwen3-VL-4B + QLoRA, grounding hư hại mặt đường).

Các module:
    common   — nhãn YOLO → JSON (0–1000 / pixel), prompt, đọc config, đường dẫn
    data     — encode_example (mask loss), Dataset, collate
    modeling — nạp model 4-bit, apply_setup (E1/E2/E3), lưu/nạp projector
    train    — vòng huấn luyện  (python -m day10.train ...)
    infer    — suy luận         (python -m day10.infer ...)
    evidence — chấm bằng scripts/score.py với --meta lấy từ run.json
    analyze  — bảng tổng hợp, ảnh minh hoạ, phân tích lỗi theo lớp, reward hacking
    reward   — reward IoU cho GRPO (E5)
    grpo     — huấn luyện GRPO  (python -m day10.grpo ...)
"""
