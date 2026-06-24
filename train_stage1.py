"""Stage 1 — Align projector. Freeze InternViT + Coder, chỉ train MLP projector.
Mục tiêu: projector học "dịch" vision embedding sang không gian Coder hiểu.
LR cao, ít step, subset nhỏ. PHẢI eval trước khi sang stage 2.

Chạy: python train_stage1.py
Ra:   out/stage1/projector.pt
"""
import torch
from transformers import Trainer, TrainingArguments
from model import MathCoderVLM
from data import make_dataset, Collator

SHARDS = "/workspace/data/math-dataset/Python_rg/*.parquet"  # reshard.py: row-group nhỏ, hết OOM

# Blackwell (sm_120): bật TF32 cho matmul/cudnn -> nhanh hơn fp32 path, không đụng bf16.
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    model = MathCoderVLM()
    model.set_trainable(projector=True, decoder_lora=False, encoder=False)  # chỉ projector
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tham số train (chỉ projector): {n_train/1e6:.1f}M")

    ds = make_dataset(SHARDS)
    args = TrainingArguments(
        output_dir="out/stage1",
        per_device_train_batch_size=4,       # seq ~2048 nặng VRAM -> batch nhỏ
        gradient_accumulation_steps=8,       # batch hiệu dụng 32
        gradient_checkpointing=True,         # đổi compute lấy VRAM (decoder full-grad, 1.5B)
        max_steps=3000,                       # align nhanh: ~96k ảnh; tăng nếu loss chưa phẳng
        learning_rate=1e-3,                   # projector mới random -> LR cao
        warmup_steps=100,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=4,            # 20 shard + row-group nhỏ -> chia worker an toàn
        dataloader_pin_memory=True,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=Collator(model))
    trainer.train()

    torch.save(model.projector.state_dict(), "out/stage1/projector.pt")
    print("OK -> out/stage1/projector.pt")


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(encoding="utf-8")
    main()
