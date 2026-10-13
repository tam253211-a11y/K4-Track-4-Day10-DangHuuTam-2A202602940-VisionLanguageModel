# Checklist Lab Day 10

## Mã (đã xong, chưa chạy GPU)
- [x] Chặng 1 — `day10/check_data.py`: dataset_version `d10-`, số ảnh/box từng split, không trùng session, 6 ảnh mẫu
- [x] Chặng 2 — `day10/common.py` (nhãn 0–1000, prompt), E0 qua `day10/infer.py --zero-shot`
- [x] Chặng 3 — `day10/data.py`: mask loss (prefix check), collate có `mm_token_type_ids`, lazy encode
- [x] Chặng 4 — `day10/modeling.py` (regex LoRA, assert 252/348, projector fp32, lưu/nạp projector), `day10/train.py`
- [x] Chặng 5 — E4 `data_size` (10% vs 100%) trên setup tốt nhất, `run_all.sh`
- [x] Chặng 6 — `day10/reward.py` + 13 test (`day10/tests/test_reward.py`), `day10/grpo.py`
- [x] Phân tích — `day10/analyze.py` (bảng, ảnh đúng/sai, lỗi theo lớp × kích thước, reward hacking), `day10/coord_check.py`

## Chạy trên GPU (Colab T4 hoặc Vast)
- [ ] `SMOKE=1 bash run_all.sh` không lỗi: 220 token ảnh, labels giải mã ra JSON + `<|im_end|>`, LoRA 252/252/348
- [ ] `report/evidence/e0.json`
- [ ] `report/evidence/e1.json` — `parse_valid_rate ≥ 0.95`, F1(E1) − F1(E0) ≥ 0.10
- [ ] `report/evidence/e2.json`, `e3.json` — có meta dương
- [ ] `report/evidence/e4_data_0.1.json`, `e4_data_1.0.json`
- [ ] (tuỳ chọn) `report/evidence/e5.json`, `report/e5_grpo_log.json`
- [ ] Tải về: `report/` (evidence, figures, runs, analysis.txt), `data/splits/meta.json` (để check_submission PASS mục meta)

## Báo cáo và nộp
- [ ] `REPORT.md`: họ tên, dataset_version, bảng (dán từ `python -m day10.analyze table`), 10 câu bằng số của mình
- [ ] Không còn chuỗi `<điền` trong REPORT.md
- [ ] ≥ 6 ảnh minh hoạ có box dự đoán + box đúng (`report/figures/<best>_*.png`) chèn vào REPORT.md
- [ ] VRAM/thời gian trong báo cáo khớp `meta` của evidence
- [ ] `git status` sạch, không có `data/`, `work/`, `.env`, `*.safetensors`, `*.pt`
- [ ] `git rev-list --count HEAD` ≥ 5
- [ ] `$env:PYTHONIOENCODING="utf-8"; python scripts/check_submission.py` → 0 FAIL
- [ ] Repo **Public**, dán URL lên LMS
