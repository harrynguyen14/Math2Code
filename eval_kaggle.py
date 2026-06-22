"""Eval Qwen3-VL math specialist trên Kaggle.
- Adapter LoRA load từ Hub: harryrobert/Qwen-3-VL-Math-Spec (base: unsloth/Qwen3-VL-8B-Instruct)
- Data parquet: /kaggle/input/data (mỗi shard = 1 loại chart)
- Metric: execution pass rate = % code sinh ra chạy được (subprocess + timeout, matplotlib Agg)
  Eval PHÂN TẦNG theo shard -> biết loại nào yếu.

Chạy trên Kaggle (bật GPU T4 15GB — model load 4-bit vừa VRAM):
    !pip install -q -U transformers peft accelerate bitsandbytes
    !python eval_kaggle.py --per-type 20
"""
import argparse, io, math, glob, os, re, subprocess, sys, tempfile
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="unsloth/Qwen3-VL-8B-Instruct")
ap.add_argument("--adapter", default="harryrobert/Qwen-3-VL-Math-Spec")
ap.add_argument("--data", default="/kaggle/input/data")
ap.add_argument("--per-type", type=int, default=20, help="số mẫu mỗi loại (mỗi shard)")
ap.add_argument("--max-new", type=int, default=3072)
ap.add_argument("--timeout", type=int, default=30)
args = ap.parse_args()

MAX_PIXELS = 1024 * 28 * 28
INSTRUCTION = "Write the Python code that reproduces the following mathematical image."

def to_pil(img):
    if isinstance(img, dict): img = img.get("bytes") or img.get("path")
    if isinstance(img, (bytes, bytearray)): img = Image.open(io.BytesIO(img))
    elif isinstance(img, str): img = Image.open(img)
    img = img.convert("RGB"); w, h = img.size
    if w * h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (w * h)); w, h = int(w * s), int(h * s)
    return img.resize((max(28, w - w % 28), max(28, h - h % 28)))

def strip_code(t):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", t, re.DOTALL)
    if m: return m.group(1).strip()
    for i, l in enumerate(t.splitlines()):
        if l.lstrip().startswith(("import ", "from ")):
            return "\n".join(t.splitlines()[i:]).strip()
    return t.strip()

PREFIX = ("import matplotlib\nmatplotlib.use('Agg')\n"
          "import matplotlib.pyplot as _p\n_p.show=lambda *a,**k:None\n")
def runs_ok(code):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(PREFIX + code); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=args.timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)

# --- load base + adapter (transformers + peft thuần, không cần Unsloth trên Kaggle) ---
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# ponytail: T4 không hỗ trợ bf16 tốt -> compute dtype fp16; NF4 4-bit để 8B vừa 15GB VRAM
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
print("Loading base + adapter (4-bit)...")
base = AutoModelForImageTextToText.from_pretrained(
    args.base, quantization_config=bnb, dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, args.adapter)
model.eval()
proc = AutoProcessor.from_pretrained(args.adapter)

# --- eval phân tầng: mỗi shard (1 loại) lấy --per-type mẫu held-out (cuối shard) ---
import pyarrow.parquet as pq

shards = sorted(glob.glob(os.path.join(args.data, "*.parquet")))
print(f"Found {len(shards)} shards (chart types)\n")

overall_pass, overall_n = 0, 0
results = {}
for shard in shards:
    name = os.path.basename(shard)
    tbl = pq.read_table(shard, columns=["image", "text"])
    total = tbl.num_rows
    take = min(args.per_type, total)
    rows = tbl.slice(max(0, total - take), take).to_pylist()  # lấy cuối shard (train đọc từ đầu)

    passed = 0
    for ex in rows:
        img = to_pil(ex["image"])
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": INSTRUCTION}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=args.max_new, do_sample=False)
        gen = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        passed += runs_ok(strip_code(gen))
    rate = passed / take
    results[name] = (passed, take, rate)
    overall_pass += passed; overall_n += take
    print(f"{name:35s} {passed:3d}/{take:3d} = {rate:.0%}")

print("\n" + "=" * 50)
print(f"OVERALL EXEC PASS RATE: {overall_pass}/{overall_n} = {overall_pass/overall_n:.1%}")
print("=" * 50)
weak = [n for n, (_, _, r) in results.items() if r < 0.8]
if weak:
    print("\nLOAI YEU (<80%, can them data/train):")
    for n in weak: print(" -", n, f"{results[n][2]:.0%}")
