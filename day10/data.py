"""Ví dụ huấn luyện: chat template + mask loss, Dataset và collate (GUIDE bước 4)."""
import torch
from PIL import Image
from torch.utils.data import Dataset

from day10.common import DATA, get_prompt, objects_to_answer

IGNORE = -100
# Các khoá văn bản cần pad; transformers 5.19 BẮT BUỘC có mm_token_type_ids để tính M-RoPE
# (thiếu sẽ báo ValueError "mm_token_type_ids is missing").
TEXT_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids")


def user_message(image, prompt):
    return [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": prompt}]}]


def encode_example(processor, image, prompt, answer):
    user = user_message(image, prompt)
    full = user + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt_enc = processor.apply_chat_template(user, add_generation_prompt=True, **kw)
    full_enc = processor.apply_chat_template(full, add_generation_prompt=False, **kw)

    n = prompt_enc["input_ids"].shape[1]
    if not torch.equal(full_enc["input_ids"][0, :n], prompt_enc["input_ids"][0]):
        raise ValueError("prompt không phải tiền tố của chuỗi đầy đủ: kiểm tra chat template")

    labels = full_enc["input_ids"].clone()
    labels[:, :n] = IGNORE                       # không học ảnh + prompt, chỉ học câu trả lời
    ex = {k: full_enc[k][0] for k in TEXT_KEYS if k in full_enc}
    ex["labels"] = labels[0]
    ex["pixel_values"] = full_enc["pixel_values"]
    ex["image_grid_thw"] = full_enc["image_grid_thw"]
    return ex


class GroundingDataset(Dataset):
    """Mã hoá lười từng ảnh trong __getitem__ (mã hoá trước 1.422 ảnh sẽ tốn nhiều GB RAM)."""

    def __init__(self, records, processor, coord="norm1000", data_dir=DATA):
        self.records, self.processor, self.coord, self.data_dir = records, processor, coord, data_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        image = Image.open(self.data_dir / r["image_path"]).convert("RGB")
        prompt = get_prompt(self.coord, r["width"], r["height"])
        answer = objects_to_answer(r["objects"], self.coord, r["width"], r["height"])
        return encode_example(self.processor, image, prompt, answer)


def make_collate(pad_token_id):
    def collate(batch):
        out = {}
        for k in TEXT_KEYS:
            if k in batch[0]:
                pad = pad_token_id if k == "input_ids" else 0
                out[k] = torch.nn.utils.rnn.pad_sequence([b[k] for b in batch], batch_first=True,
                                                         padding_value=pad)
        out["labels"] = torch.nn.utils.rnn.pad_sequence([b["labels"] for b in batch], batch_first=True,
                                                        padding_value=IGNORE)
        # tensor ảnh: nối theo chiều 0, KHÔNG pad (mỗi ảnh là một khối patch riêng, image_grid_thw chỉ ranh giới)
        out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
        out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
        return out
    return collate


def num_image_tokens(image_grid_thw, merge_size=2):
    """Số token ảnh đưa vào LLM = t·h·w / merge² (Qwen3-VL: patch 16, gộp 2×2 → 32 px/token)."""
    return int(image_grid_thw.prod(-1).sum().item()) // merge_size ** 2
