from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import glob


max_seq_length = 4096 
model_name = "Qwen/Qwen3-VL-8B-Instruct"


model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=False,
    max_seq_length=max_seq_length,
    use_gradient_checkpointing=False,           # tắt: GPU đủ VRAM, train nhanh hơn ~10-20%
)

# Giới hạn token ảnh: ảnh dataset rộng tới 3000px sẽ ngốn token khổng lồ + phình VRAM.
# 1024*28*28 ~ 800k px (vd 896x896) đủ chi tiết cho hình toán mà giữ chuỗi trong max_seq_length.
# Resize ảnh thủ công trong to_messages thay vì vá processor internals
# (vá __dict__ trên transformers 5.x làm patch count lệch pos_embed -> RuntimeError shape mismatch ở vision tower).
MAX_PIXELS = 1024 * 28 * 28


model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,                
    finetune_language_layers=True,              
    finetune_attention_modules=True,            
    finetune_mlp_modules=True,                  
    r=64,                                       # capacity lớn để model "ngấm" thành specialist image->Python
    lora_alpha=128,                             # alpha = 2*r, scale mạnh hơn cho chuyên biệt hoá
    lora_dropout=0,
    bias="none",
    random_state=3407,
)


# MỖI SHARD = 1 LOẠI CHART khác nhau (shard0=hình, shard1=biểu đồ, shard2=chart, ...).
# Container giới hạn 48GB RAM -> interleave 20 stream song song hoặc buffer to = OOM Killed.
# 1 stream gộp 20 file + shuffle buffer vừa phải (ảnh decode nặng RAM nên không để quá lớn).
shards = sorted(glob.glob("/data/math-dataset/Python/*.parquet"))
dataset = load_dataset("parquet", data_files={"train": shards}, split="train", streaming=True)
dataset = dataset.shuffle(seed=3407, buffer_size=3000)


INSTRUCTION = "Write the Python code that reproduces the following mathematical image."

import math, io
from PIL import Image

def _resize(img):
    # cột parquet có thể là bytes thô, dict {"bytes":...}, hoặc PIL Image -> chuẩn hoá về PIL
    if isinstance(img, dict):
        img = img.get("bytes") or img.get("path")
    if isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img))
    elif isinstance(img, str):
        img = Image.open(img)
    # giảm ảnh về <= MAX_PIXELS, giữ tỉ lệ; bội số 28 (patch size) để grid khớp pos_embed
    img = img.convert("RGB")
    w, h = img.size
    if w * h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (w * h))
        w, h = int(w * s), int(h * s)
    w = max(28, w - w % 28)
    h = max(28, h - h % 28)
    return img.resize((w, h))

def to_messages(batch):
    # batched transform on-the-fly: KHÔNG ghi cache ra đĩa (tránh hết dung lượng)
    return {"messages": [
        [
            {"role": "user", "content": [
                {"type": "image", "image": _resize(img)},
                {"type": "text", "text": INSTRUCTION},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": txt},
            ]},
        ]
        for img, txt in zip(batch["image"], batch["text"])
    ]}

# streaming: .map chạy on-the-fly, không ghi cache ra đĩa
dataset = dataset.map(to_messages, batched=True, remove_columns=["id", "image", "text", "source"])


training_args = SFTConfig(
    per_device_train_batch_size=1,              # PHẢI =1: Unsloth Qwen3-VL crash khi batch nhiều ảnh khác grid_thw (pos_embeds shape mismatch)
    gradient_accumulation_steps=16,             # bù lại batch=1 -> tổng batch vẫn 16
    warmup_steps=100,
    max_steps=16000,                            # 16000*16 = 256k ảnh (~13k/loại), ~24h. streaming -> dùng max_steps thay epoch. Tăng nếu eval còn yếu
    learning_rate=2e-4,
    bf16=True,                                 
    logging_steps=1,
    optim="adamw_8bit",                         # tiết kiệm VRAM lớn, gần như không ảnh hưởng chất lượng với LoRA
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    output_dir="outputs_qwen3_math_5090",
    save_strategy="steps",                      # train nhiều ngày -> checkpoint để không mất tiến độ khi crash
    save_steps=2000,                            # train ~50h -> giãn save (125k steps); save_total_limit giữ 3 cái mới nhất
    save_total_limit=3,

    remove_unused_columns=False,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
)


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(
        model, tokenizer,
        train_on_responses_only=True,           # chỉ tính loss trên code (assistant), dồn capacity vào sinh code
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    ),
    train_dataset=dataset,
    args=training_args,
)

# 7. Tiến hành huấn luyện
print("Đang khởi động tiến trình huấn luyện LoRA BF16 trên GPU RTX 5090...")
trainer_stats = trainer.train()

# 8. Lưu adapter LoRA (nhẹ, để resume/thử nghiệm)
model.save_pretrained("qwen3_vl_math_lora_5090_weights")
tokenizer.save_pretrained("qwen3_vl_math_lora_5090_weights")

# 9. Merge LoRA vào base -> mô hình specialist 16-bit độc lập, deploy trực tiếp không cần adapter
model.save_pretrained_merged(
    "qwen3_vl_math_specialist_16bit",
    tokenizer,
    save_method="merged_16bit",
)
print("Hoàn thành! Adapter ở 'qwen3_vl_math_lora_5090_weights', model specialist 16-bit ở 'qwen3_vl_math_specialist_16bit'")
