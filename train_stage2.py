"""Stage 2 — SFT chuyên biệt hóa. Load projector đã align, gắn LoRA vào Coder,
train projector + LoRA decoder trên full 4tr ảnh. InternViT freeze.

Chạy: python train_stage2.py
Ra:   out/stage2 (checkpoint) + projector cuối
"""
import os
# gộp segment phân mảnh: step 1855 OOM với 5.77GB "reserved but unallocated" -> fragment.
# Phải set TRƯỚC import torch. Cho seq dài (code to) khỏi vỡ ở đỉnh VRAM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from model import MathCoderVLM
from data import make_dataset, Collator

SHARDS = "/workspace/data/math-dataset/Python_rg/*.parquet"  # reshard.py: row-group nhỏ, hết OOM
PROJECTOR = "out/stage1/projector.pt"

# Blackwell (sm_120): TF32 matmul/cudnn.
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    model = MathCoderVLM()
    sd = torch.load(PROJECTOR, map_location="cpu")  # projector đã align từ stage 1
    model.projector.load_state_dict(sd)
    print(f"Loaded projector từ {PROJECTOR}")

    # LoRA chỉ trên decoder (Coder); encoder freeze; projector train full
    lora = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model.decoder = get_peft_model(model.decoder, lora)
    # seq 4096 gấp đôi activation -> bật grad-ckp decoder để vừa VRAM (đổi compute lấy bộ nhớ).
    model.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.decoder.enable_input_require_grads()  # cần cho grad-ckp khi input là inputs_embeds
    for p in model.encoder.parameters(): p.requires_grad_(False)
    for p in model.projector.parameters(): p.requires_grad_(True)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tham số train (projector + LoRA): {n/1e6:.1f}M")

    ds = make_dataset(SHARDS)
    args = TrainingArguments(
        output_dir="out/stage2",
        per_device_train_batch_size=8,       # group_by_length + grad-ckp; đỉnh = batch toàn seq 4096
        gradient_accumulation_steps=4,       # batch hiệu dụng 32
        max_steps=30000,                      # ~960k ảnh; tăng dần, eval giữa chừng (đừng cam kết mù 4tr)
        learning_rate=2e-4,
        warmup_steps=200,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",           # fused optim: nhanh hơn trên CUDA
        weight_decay=0.01,
        logging_steps=10,
        # save đầy đủ (optimizer+scheduler+data) để resume thật. save_safetensors=False -> torch.save
        # (pickle) xử được tied-weight Qwen2 (embed/lm_head share), safetensors thì raise.
        save_strategy="steps",
        save_steps=1000,                     # dày để OOM/crash chỉ mất <1000 step; resume = chạy lại lệnh
        save_total_limit=3,
        save_safetensors=False,
        remove_unused_columns=False,
        group_by_length=True,                # gom mẫu cùng độ dài/batch -> ít pad phí, VRAM phẳng
        length_column_name="length",         # cột thêm trong make_dataset (len(text), ~token)
        dataloader_num_workers=4,            # 20 shard + row-group nhỏ -> chia worker an toàn
        dataloader_pin_memory=True,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=Collator(model))
    # resume nếu có checkpoint cũ trong out/stage2 (stop giữa chừng -> chạy lại lệnh là tiếp)
    ckpts = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")] \
        if os.path.isdir(args.output_dir) else []
    trainer.train(resume_from_checkpoint=bool(ckpts))

    model.save_pretrained("out/stage2/final")   # 1 thư mục: decoder(LoRA)+projector+tokenizer
    print("OK -> out/stage2/final (load lại: MathCoderVLM.from_pretrained('out/stage2/final'))")


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(encoding="utf-8")
    main()
