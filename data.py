"""Dataset streaming + collator dùng chung cho train_stage1/2.

FigCodifier parquet: cột `image` (bytes/dict/PIL) + `text` (code Python).
Mỗi mẫu -> prompt có N image token (N = số vision token sau encode) + code làm label.
Resize ảnh về 448x448 (InternViT single-tile; tiling đa ô để sau nếu cần độ phân giải cao hơn).
"""
import io, glob, os
import torch
from PIL import Image
from datasets import load_dataset

# ~ thường nhỏ; trỏ cache Arrow sang ổ workspace (to hơn) để khỏi đầy đĩa khi build 2.7M dòng.
# ponytail: override bằng HF_DATASETS_CACHE nếu ổ khác; chỉ set khi chưa có sẵn.
os.environ.setdefault("HF_DATASETS_CACHE", "/workspace/hf_cache")

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
    """Non-streaming: load shard (~50k dòng/file từ reshard.py) -> Arrow memory-map (ko nạp
    hết ảnh vào RAM). Random-access được -> Trainer group_by_length gom mẫu cùng độ dài,
    VRAM phẳng + ít pad phí. shuffle toàn bộ (random-access nên trộn thật, ko cần buffer).
    Lần đầu datasets build Arrow cache 1 lần (ngốn đĩa); các lần sau load nhanh.
    """
    shards = sorted(glob.glob(shards_glob))
    assert shards, f"Không thấy parquet ở {shards_glob}"
    ds = load_dataset("parquet", data_files=shards, split="train")  # non-streaming
    # cột length cho group_by_length: xấp xỉ token bằng len(text) (~tỉ lệ) -> đủ để gom batch cùng cỡ.
    # ds["text"] đọc qua HF Arrow vẫn quét block ảnh -> treo trên 4.1M dòng. Đọc THẲNG cột text
    # từ parquet bằng pyarrow (column-prune thật, ko đụng cột ảnh) -> nhanh, ko nhân đôi đĩa.
    import pyarrow.parquet as pq
    from tqdm import tqdm
    lengths = []
    for f in tqdm(shards, desc="length (cho group_by_length)"):
        col = pq.read_table(f, columns=["text"]).column("text")
        lengths.extend(len(s) for s in col.to_pylist())
    ds = ds.add_column("length", lengths)
    return ds.shuffle(seed=seed)


class Collator:
    """Gom batch -> input_ids/attention_mask/pixel_values/labels.
    train_on_responses_only: mask label phần prompt = -100, chỉ tính loss trên code.
    """
    def __init__(self, model, max_len=4096):  # 4096 mới bao hết code dài nhất (đã analyze); 2048 cắt cụt
        self.tok = model.tok
        self.image_token = "<|image_pad|>"
        self.image_proc = model.image_processor
        self.max_len = max_len

    def __call__(self, batch):
        # xử lý ảnh theo cả batch 1 lần (vectorized) thay vì từng ảnh -> nhanh hơn vòng lặp
        imgs = [_to_pil(ex["image"]) for ex in batch]
        pixels = self.image_proc(imgs, return_tensors="pt").pixel_values  # [B,3,H,W]

        input_ids, labels, masks = [], [], []
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
            masks.append(torch.ones(len(ids), dtype=torch.long))  # mask theo ĐỘ DÀI thật

        # pad=eos (Qwen ko có pad riêng) -> KO tính mask bằng (ids!=pad): sẽ nuốt luôn eos cuối câu
        # => model ko học token dừng. Pad mask riêng (giá trị 0) để eos cuối giữ mask=1.
        pad = self.tok.pad_token_id or self.tok.eos_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attn = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
        return {
            "input_ids": input_ids, "attention_mask": attn,
            "labels": labels, "pixel_values": pixels,
        }
