"""Dataset streaming + collator dùng chung cho train_stage1/2.

FigCodifier parquet: cột `image` (bytes/dict/PIL) + `text` (code Python).
Mỗi mẫu -> prompt có N image token (N = số vision token sau encode) + code làm label.
Resize ảnh về 448x448 (InternViT single-tile; tiling đa ô để sau nếu cần độ phân giải cao hơn).
"""
import io, glob
import torch
from PIL import Image
from datasets import load_dataset, interleave_datasets

INSTRUCTION = "Write the Python code that reproduces the following mathematical image."
IMG_SIZE = 448
N_VIS = (IMG_SIZE // 14) ** 2 // 4  # patch 14 -> 1024, pixel_shuffle 2x2 -> 256 tokens/ảnh


def _to_pil(img):
    if isinstance(img, dict):
        img = img.get("bytes") or img.get("path")
    if isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img))
    elif isinstance(img, str):
        img = Image.open(img)
    return img.convert("RGB")  # resize để image_processor lo (tránh resize 2 lần)


def make_dataset(shards_glob, seed=3407, buffer=10000):
    """Mỗi shard = 1 category (shard0 hình học, shard1 chart...). Nếu đọc tuần tự
    sẽ train hết category này mới sang category kia -> catastrophic forgetting.
    interleave_datasets xoay vòng các shard -> mỗi batch lẫn đủ category.
    shuffle buffer to (RAM 64GB dư) để trộn mịn thêm trong từng stream.
    """
    shards = sorted(glob.glob(shards_glob))
    assert shards, f"Không thấy parquet ở {shards_glob}"
    streams = [
        load_dataset("parquet", data_files=s, split="train", streaming=True)
        for s in shards
    ]
    ds = interleave_datasets(streams, seed=seed)  # xoay vòng đều giữa các shard
    return ds.shuffle(seed=seed, buffer_size=buffer)


class Collator:
    """Gom batch -> input_ids/attention_mask/pixel_values/labels.
    train_on_responses_only: mask label phần prompt = -100, chỉ tính loss trên code.
    """
    def __init__(self, model, max_len=2048):
        self.tok = model.tok
        self.image_token = "<|image_pad|>"
        self.image_proc = model.image_processor
        self.max_len = max_len

    def __call__(self, batch):
        # xử lý ảnh theo cả batch 1 lần (vectorized) thay vì từng ảnh -> nhanh hơn vòng lặp
        imgs = [_to_pil(ex["image"]) for ex in batch]
        pixels = self.image_proc(imgs, return_tensors="pt").pixel_values  # [B,3,H,W]

        input_ids, labels = [], []
        for ex in batch:
            prompt = (f"<|im_start|>user\n{self.image_token * N_VIS}{INSTRUCTION}"
                      f"<|im_end|>\n<|im_start|>assistant\n")
            answer = ex["text"] + self.tok.eos_token
            p_ids = self.tok(prompt, add_special_tokens=False).input_ids
            a_ids = self.tok(answer, add_special_tokens=False).input_ids
            ids = (p_ids + a_ids)[: self.max_len]
            lab = ([-100] * len(p_ids) + a_ids)[: self.max_len]  # chỉ loss trên code
            input_ids.append(torch.tensor(ids))
            labels.append(torch.tensor(lab))

        pad = self.tok.pad_token_id or self.tok.eos_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attn = (input_ids != pad).long()
        return {
            "input_ids": input_ids, "attention_mask": attn,
            "labels": labels, "pixel_values": pixels,
        }
