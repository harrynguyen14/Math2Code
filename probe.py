"""Probe: in shape các tensor collator tạo. Chạy: python probe.py
Nếu pixel_values KHÔNG phải 2-D [tổng_patches, feat] hoặc image_grid_thw thiếu -> collator/processor sai dạng."""
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from datasets import load_dataset
import math, io
from PIL import Image

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    load_in_4bit=False, max_seq_length=4096,
    use_gradient_checkpointing="unsloth",
)
FastVisionModel.for_training(model)

MAX_PIXELS = 1024 * 28 * 28
def _resize(img):
    if isinstance(img, dict): img = img.get("bytes") or img.get("path")
    if isinstance(img, (bytes, bytearray)): img = Image.open(io.BytesIO(img))
    elif isinstance(img, str): img = Image.open(img)
    img = img.convert("RGB")
    w, h = img.size
    if w*h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS/(w*h)); w, h = int(w*s), int(h*s)
    return img.resize((max(28, w-w%28), max(28, h-h%28)))

ds = load_dataset("parquet", data_files={"train":"/data/math-dataset/Python/*.parquet"},
                  split="train", streaming=True)
INSTRUCTION = "Write the Python code that reproduces the following mathematical image."
samples = []
for ex in ds:
    samples.append({"messages":[
        {"role":"user","content":[{"type":"image","image":_resize(ex["image"])},{"type":"text","text":INSTRUCTION}]},
        {"role":"assistant","content":[{"type":"text","text":ex["text"]}]},
    ]})
    if len(samples) == 4: break

collator = UnslothVisionDataCollator(
    model, tokenizer, train_on_responses_only=True,
    instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n",
)
batch = collator(samples)
print("=== keys ===", list(batch.keys()))
for k, v in batch.items():
    try: print(f"{k:20s} shape={tuple(v.shape)} dtype={v.dtype}")
    except Exception: print(f"{k:20s} = {v}")
if "image_grid_thw" in batch:
    print("image_grid_thw =", batch["image_grid_thw"])
    print("sum(t*h*w) =", int(batch["image_grid_thw"].prod(dim=1).sum()))
