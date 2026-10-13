"""Nạp model 4-bit, setup LoRA/đóng băng cho E1–E3, lưu/nạp projector (GUIDE bước 3, 5, 7)."""
from pathlib import Path

import torch

from day10.common import MODEL_ID, read_json

# Tên module lấy từ cây model thật (transformers 5.19, Qwen3VLForConditionalGeneration):
#   model.language_model.layers.N.self_attn.{q,k,v,o}_proj, .mlp.{gate,up,down}_proj   → 36 × 7 = 252
#   model.visual.blocks.N.attn.{qkv,proj}, .mlp.{linear_fc1,linear_fc2}                 → 24 × 4 = 96
# Dùng regex (PEFT fullmatch) thay vì list hậu tố: list ["linear_fc1","linear_fc2"] sẽ khớp nhầm cả
# visual.merger.linear_fc* và visual.deepstack_merger_list.N.linear_fc* (projector), còn thiếu attn.proj.
LLM_RE = r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
VIT_RE = r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.linear_fc[12])"
PROJ_KEYS = ("visual.merger.", "visual.deepstack_merger_list.")
EXPECTED_LORA = {"llm": 252, "llm+vit": 348}


def quant_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        # giữ ViT (kể cả merger/deepstack) và lm_head ở fp16: nếu ViT bị lượng tử hoá 4-bit,
        # trọng số merger thành Params4bit (uint8 đóng gói) → không thể huấn luyện đầy đủ ở E2/E3.
        # transformers 5.19 so khớp bằng re.match(key, full_name) — neo ở ĐẦU tên đầy đủ — nên "visual"
        # (như GUIDE) KHÔNG khớp "model.visual.blocks..." và ViT bị lượng tử hoá âm thầm. Phải ghi "model.visual".
        llm_int8_skip_modules=["model.visual", "lm_head"],
    )


def assert_vit_not_quantized(model):
    """ViT/projector phải là nn.Linear fp16 thường; LLM phải là Linear4bit."""
    import bitsandbytes as bnb
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    vis = [(n, m) for n, m in base.named_modules() if ".visual." in f".{n}." and isinstance(m, torch.nn.Linear)]
    q_vis = [n for n, m in vis if isinstance(m, bnb.nn.Linear4bit)]
    q_llm = sum(isinstance(m, bnb.nn.Linear4bit) for n, m in base.named_modules() if "language_model" in n)
    if q_vis or not q_llm:
        raise RuntimeError(f"lượng tử hoá sai: {len(q_vis)} Linear của ViT bị 4-bit (vd {q_vis[:2]}), "
                           f"{q_llm} Linear4bit trong LLM")
    print(f"[quant] LLM: {q_llm} Linear4bit · ViT+projector: {len(vis)} nn.Linear "
          f"{base.model.visual.merger.linear_fc1.weight.dtype} (không lượng tử hoá)")


def load_processor(model_id=MODEL_ID):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"      # cho generate theo batch
    return processor


def load_base(model_id=MODEL_ID):
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant_config(), dtype=torch.float16, device_map={"": 0})
    assert_vit_not_quantized(model)
    return model


def is_projector(name):
    return any(k in name for k in PROJ_KEYS)


def count_lora_modules(model):
    from peft.tuners.lora import LoraLayer
    return sum(1 for _, m in model.named_modules() if isinstance(m, LoraLayer))


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_setup(model, targets="llm", train_projector=False, r=16, alpha=32, dropout=0.05,
                gradient_checkpointing=True):
    """E1: targets=llm. E2: + train_projector. E3: targets=llm+vit + train_projector."""
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()        # cần để gradient đi qua các lớp checkpoint
    model.config.use_cache = False

    pattern = LLM_RE if targets == "llm" else f"(?:{LLM_RE})|(?:{VIT_RE})"
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=pattern,
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)

    n_lora = count_lora_modules(model)
    if n_lora != EXPECTED_LORA[targets]:
        raise RuntimeError(f"số module LoRA = {n_lora}, kỳ vọng {EXPECTED_LORA[targets]} ({targets})")

    lora_params = trainable_params(model)
    proj_params = 0
    if train_projector:
        for n, p in model.named_parameters():
            if is_projector(n):
                # fp32: master weight cho Adam; cập nhật lr·grad (~1e-7) bị mất khi cộng vào trọng số fp16
                p.data = p.data.to(torch.float32)
                p.requires_grad = True
                proj_params += p.numel()

    bad = [n for n, p in model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    if bad:  # GradScaler.unscale_ không chấp nhận gradient fp16
        raise RuntimeError(f"tham số huấn luyện không phải fp32: {bad[:3]}")

    info = {"lora_modules": n_lora, "lora_params": lora_params, "projector_params": proj_params,
            "trainable_params": trainable_params(model)}
    print(f"[apply_setup] targets={targets} projector={train_projector} r={r} alpha={alpha} → {info}")
    return model, info


def projector_state(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return {n: p.detach().to("cpu", copy=True) for n, p in base.named_parameters() if is_projector(n)}


def save_projector(model, path):
    """peft.save_pretrained() CHỈ lưu adapter LoRA → projector huấn luyện đầy đủ phải lưu riêng."""
    sd = projector_state(model)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, path)
    return len(sd)


def load_projector(model, path):
    sd = torch.load(path, map_location="cpu")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    params = dict(base.named_parameters())
    unexpected = [k for k in sd if k not in params]
    if unexpected or not sd:
        raise RuntimeError(f"projector không khớp model: {unexpected[:3]} (n={len(sd)})")
    k0 = next(iter(sd))
    before = params[k0].detach().float().sum().item()
    with torch.no_grad():
        for k, v in sd.items():
            params[k].data.copy_(v.to(params[k].dtype))
    after = params[k0].detach().float().sum().item()
    print(f"[load_projector] nạp {len(sd)} tensor; checksum {k0}: {before:.4f} → {after:.4f}")
    if before == after:
        print("  CẢNH BÁO: projector không đổi sau khi nạp — kết quả E2/E3 sẽ giống E1")
    return len(sd)


def load_trained(run_dir, is_trainable=False):
    """Nạp model gốc 4-bit → adapter LoRA → projector (nếu có). Dùng cho suy luận và khởi tạo GRPO."""
    from peft import PeftModel
    run_dir = Path(run_dir)
    run = read_json(run_dir / "run.json")
    model = load_base(run.get("model_id", MODEL_ID))
    model = PeftModel.from_pretrained(model, str(run_dir / "adapter"), is_trainable=is_trainable)
    if run.get("train_projector"):
        load_projector(model, run_dir / "projector.pt")
    return model, run


def architecture_breakdown(model_id=MODEL_ID):
    """Đếm tham số từng khối trên model 'meta' (không tải trọng số, không lượng tử hoá) — cho Câu 1."""
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    cfg = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        m = Qwen3VLForConditionalGeneration._from_config(cfg)
    groups = {"vit_blocks+embed": 0, "visual.merger": 0, "visual.deepstack_merger_list": 0,
              "language_model": 0, "lm_head(untied)": 0}
    for n, p in m.named_parameters():   # named_parameters() bỏ trùng tham số tied (lm_head = embed_tokens)
        if "visual.merger." in n:
            groups["visual.merger"] += p.numel()
        elif "visual.deepstack_merger_list." in n:
            groups["visual.deepstack_merger_list"] += p.numel()
        elif ".visual." in f".{n}":
            groups["vit_blocks+embed"] += p.numel()
        elif "language_model" in n:
            groups["language_model"] += p.numel()
        else:
            groups["lm_head(untied)"] += p.numel()
    groups["total"] = sum(groups.values())
    groups["tie_word_embeddings"] = bool(getattr(cfg.text_config, "tie_word_embeddings",
                                                 getattr(cfg, "tie_word_embeddings", False)))
    vc = cfg.vision_config
    groups["vision"] = {"patch_size": vc.patch_size, "spatial_merge_size": vc.spatial_merge_size,
                        "depth": vc.depth, "deepstack_visual_indexes": list(vc.deepstack_visual_indexes)}
    groups["text_layers"] = cfg.text_config.num_hidden_layers
    return groups
