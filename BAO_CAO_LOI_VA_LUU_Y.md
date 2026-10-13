# Báo cáo lỗi và lưu ý kỹ thuật — Lab Day 10

Các bẫy kỹ thuật mình gặp hoặc phát hiện khi đối chiếu pseudo code/đoạn mã mẫu của đề với **mã nguồn thật** của các
phiên bản đã ghim (`transformers==5.19.0`, `peft==0.21.2`, `trl==1.9.2`). Mỗi mục ghi: triệu chứng → nguyên nhân → cách
xử lý trong repo này.

## 1. Collate mẫu thiếu `mm_token_type_ids` → lỗi ngay ở bước forward
- **Triệu chứng:** `ValueError: Multimodal data was passed (via image_grid_thw ...) but mm_token_type_ids is missing`.
- **Nguyên nhân:** `Qwen3VLModel` (transformers 5.19) cần `mm_token_type_ids` để tính M-RoPE 3D cho token ảnh; processor
  trả khoá này cùng `input_ids` nhưng đoạn `collate_fn` mẫu chỉ pad `input_ids/attention_mask/labels`.
- **Xử lý:** `day10/data.py` giữ `mm_token_type_ids` trong `encode_example` và pad bằng 0 trong `collate` (GUIDE bước 4
  cũng ghi rõ khoá này).

## 2. `target_modules` dạng list hậu tố khớp nhầm projector ở E3
- **Nguyên nhân:** PEFT so khớp list theo hậu tố tên module. `["qkv", "linear_fc1", "linear_fc2"]` khớp cả
  `visual.merger.linear_fc1/2` và `visual.deepstack_merger_list.N.linear_fc1/2` (projector cũng có hai lớp tên như vậy),
  đồng thời **thiếu** `attn.proj` của ViT → số module ≠ 348, projector vừa bị bọc LoRA vừa được mở full, và khoá
  state dict của projector bị đổi thành `...base_layer.weight`.
- **Đo trên cây module thật** (model `meta`, không tải trọng số): list hậu tố của đề cho **332** module LoRA, trong đó
  **8** module nằm trong merger/deepstack_merger; regex cho đúng **252 / 252 / 348** và 0 module trên projector.
- **Xử lý:** dùng regex trên tên đầy đủ (PEFT `re.fullmatch`) và `assert` số module = 252 (E1/E2) / 348 (E3) —
  `day10/modeling.py::LLM_RE, VIT_RE`. Số tham số huấn luyện đo được (r=16): E1 33.030.144 · E2 142.135.296
  (+109.105.152 projector) · E3 148.426.752.

## 3. "Patch 14×14" là của Qwen2.5-VL, không phải Qwen3-VL
- Qwen3-VL dùng `patch_size=16`, `spatial_merge_size=2` → mỗi token ảnh ứng với 32×32 px. Ảnh 640×360 được resize về
  bội số của 32: 640×352 → lưới 40×22 patch → 20×11 = **220 token**. (Với patch 14 sẽ ra ~299 token, không khớp 220.)
- Kiểm chứng: `image_grid_thw` của một ví dụ và `num_image_tokens()` ở notebook mục 4.

## 4. `GRPOConfig` mặc định bf16 — T4 không hỗ trợ
- trl 1.9.2: "`bf16` defaults to True if `fp16` is not set". T4 (sm75) không có bf16 phần cứng.
- **Xử lý:** `fp16=True, bf16=False` trong `day10/grpo.py`.

## 5. GRPOTrainer ép tham số huấn luyện của model 4-bit sang bf16
- `GRPOTrainer.__init__` (trl 1.9.2): với model lượng tử hoá, mọi tham số `requires_grad` bị `param.data.to(bfloat16)`
  (theo khuyến nghị QLoRA). Trên T4 + fp16 AMP điều này vừa mất độ chính xác adapter SFT vừa xung đột GradScaler.
- **Xử lý:** lưu bản fp32 của tham số LoRA trước khi tạo trainer rồi khôi phục ngay sau đó (optimizer chỉ được tạo
  trong `train()`), có `assert` kiểu fp32.

## 6. `warmup_ratio` không còn trong `TrainingArguments` của transformers 5.19
- Dùng `warmup_steps` (nhận int hoặc float tỉ lệ).

## 7. Projector phải lưu riêng và nạp lại có kiểm chứng
- `peft_model.save_pretrained()` chỉ lưu adapter LoRA; projector E2/E3 huấn luyện đầy đủ nên lưu `projector.pt`.
- Khi nạp: đối chiếu tên với `get_base_model().named_parameters()`, báo lỗi nếu có khoá lạ, và in checksum trước/sau để
  chắc projector đã thay đổi (nếu không, E2/E3 sẽ cho kết quả y hệt E1).
- Projector huấn luyện ở fp32 (cần `autocast` fp16 khi forward); lúc suy luận được chép về tham số fp16 của model gốc.

## 8. Số `--meta` trong đoạn mẫu của đề chỉ là ví dụ
- Ví dụ `trainable_params=28311552` không khớp cấu hình thật: LoRA r=16 trên 252 module của Qwen3-4B
  (hidden 2560, MLP 9728, q 4096, kv 1024, 36 lớp) là 36 × 917.504 = **33.030.144** tham số.
- **Xử lý:** `day10/evidence.py` đọc meta nguyên văn từ `run.json` do vòng train ghi ra, không gõ tay.

## 9. Mâu thuẫn giữa test của đề và `check_submission.py` về chuỗi `<điền`
- `tests/test_configs_and_docs.py::test_report_template_has_student_markers...` yêu cầu REPORT.md **còn** `<điền`;
  `check_submission.py` báo FAIL nếu **còn** `<điền`. Rubric chấm bằng `check_submission.py` → điền hết; test kia sẽ
  fail sau khi điền báo cáo. Không sửa `tests/` (−10 điểm).

## 10. Box suy biến bị trọng tài coi là JSON hỏng
- `score.parse_output` trả `None` (hỏng cả ảnh) nếu một box có `x2 <= x1` hoặc `y2 <= y1`. Box rất nhỏ (diện tích
  trung vị ≈ 0,7% khung hình) khi làm tròn có thể suy biến → `yolo_to_box` nới ra 1 đơn vị, kể cả sát mép 1000.

## 11. Môi trường
- `trl==1.9.2` import `requests`, nhưng `huggingface_hub` 2.x không còn kéo theo gói này → `No module named 'requests'`
  khi `from trl import GRPOTrainer` trên máy sạch. Đã thêm `requests` vào lệnh cài.
- Colab: `SecretNotFoundError: Secret KAGGLE_USERNAME does not exist` → tạo secret đúng tên và bật **Notebook access**;
  hoặc tải zip tay và `prepare_data.py --zip`.
- Windows: chạy `check_submission.py` cần `$env:PYTHONIOENCODING="utf-8"` (console cp1252).
- Máy local Python 3.13 nằm ngoài khoảng 3.10–3.12 của đề → chạy pipeline trên Colab/Vast.

## 12. Thiết kế thí nghiệm
- E4 `data_size` giữ nguyên 300 bước: 10% dữ liệu (142 ảnh) lặp ~17 epoch so với ~1,7 epoch của 100% — hiệu ứng
  "ít dữ liệu" lẫn với "overfit"; đây là hệ quả bắt buộc của việc chỉ đổi một biến.
- Reward E5 có sàn 0.2 cho mọi JSON hợp lệ và 1.0 cho `[]` trên ảnh rỗng → cần đo tỉ lệ output `[]` sau GRPO.
- Reward thưa: box nhỏ khó đạt IoU ≥ 0,5 nên cả 4 lần sinh của một ảnh thường cùng reward → advantage = 0, không có
  gradient. trl log sẵn `frac_reward_zero_std` (tỉ lệ nhóm như vậy) trong `report/e5_grpo_log.json` — dùng để phân tích.

## 13. Cách mình kiểm chứng mã trước khi thuê GPU
- Môi trường CPU với đúng các phiên bản ghim: 68 test của đề + 22 test của mình (`day10/tests`) đều pass.
- Processor thật của Qwen3-VL-4B: `image_grid_thw=[1,22,40]` → 220 token ảnh; ví dụ 371 token chỉ học 59 token;
  phần `labels != -100` giải mã đúng JSON đáp án + `<|im_end|>\n`.
- Qwen3-VL tí hon khởi tạo ngẫu nhiên (cùng kiến trúc, ít lớp): forward/backward với batch thật cho E1/E2/E3, mọi tham
  số huấn luyện có gradient, lưu → nạp adapter + projector cho logits trùng tuyệt đối (Δ = 0), generate theo batch chạy
  được; GRPO 2 bước với đúng schema dataset/reward.
- Chưa kiểm được trên CPU (cần GPU): lượng tử hoá bitsandbytes 4-bit, autocast/GradScaler CUDA, VRAM và thời gian thật
  → chạy `SMOKE=1 bash run_all.sh` trước khi chạy thật.
