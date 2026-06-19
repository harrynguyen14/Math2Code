"""Eval model image->Python sau finetune.
Metric: execution pass rate = % code sinh ra chạy được không lỗi (proxy thực tế cho chất lượng;
val loss generative gần như vô dụng). Mỗi code chạy trong subprocess riêng (sandbox + timeout),
matplotlib dùng backend 'Agg' nên không bật cửa sổ.

Chạy:  python eval.py                       # 100 mẫu held-out, model merged 16bit
       python eval.py --n 200 --adapter    # 200 mẫu, dùng adapter LoRA thay vì merged
       python eval.py --save-bad bad.jsonl # lưu các code lỗi để xem
"""
import argparse, io, math, json, subprocess, sys, tempfile, os
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen3_vl_math_specialist_16bit", help="model merged 16bit (mặc định)")
ap.add_argument("--adapter", action="store_true", help="dùng adapter LoRA (base + lora) thay vì merged")
ap.add_argument("--base", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--lora", default="qwen3_vl_math_lora_5090_weights")
ap.add_argument("--data", default="/data/math-dataset/Python/*.parquet")
ap.add_argument("--n", type=int, default=100, help="số mẫu held-out")
ap.add_argument("--seed", type=int, default=9999, help="seed KHÁC seed train (3407) -> mẫu model gần như chưa thấy")
ap.add_argument("--max-new", type=int, default=2048)
ap.add_argument("--timeout", type=int, default=30, help="giây tối đa cho mỗi code")
ap.add_argument("--save-bad", default=None, help="file jsonl lưu code lỗi")
args = ap.parse_args()

MAX_PIXELS = 1024 * 28 * 28
INSTRUCTION = "Write the Python code that reproduces the following mathematical image."

def to_pil(img):
    if isinstance(img, dict): img = img.get("bytes") or img.get("path")
    if isinstance(img, (bytes, bytearray)): img = Image.open(io.BytesIO(img))
    elif isinstance(img, str): img = Image.open(img)
    img = img.convert("RGB")
    w, h = img.size
    if w * h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (w * h)); w, h = int(w * s), int(h * s)
    return img.resize((max(28, w - w % 28), max(28, h - h % 28)))

def strip_code(text):
    # model có thể bọc ```python ... ``` -> lấy phần trong fence nếu có
    if "```" in text:
        for p in text.split("```"):
            p = p.strip()
            if p.startswith("python"): p = p[len("python"):]
            if p: return p.strip()
    return text.strip()

# chạy headless: chèn backend Agg + chặn plt.show treo tiến trình
SANDBOX_PREFIX = (
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as _plt\n"
    "_plt.show = lambda *a, **k: None\n"
)

def runs_ok(code):
    src = SANDBOX_PREFIX + code
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(src); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=args.timeout)
        return (r.returncode == 0), (r.stderr or "")[-500:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        os.unlink(path)

# --- load model ---
from unsloth import FastVisionModel
import torch

if args.adapter:
    model, tokenizer = FastVisionModel.from_pretrained(args.base, load_in_4bit=False, max_seq_length=4096)
    model.load_adapter(args.lora)
else:
    model, tokenizer = FastVisionModel.from_pretrained(args.model, load_in_4bit=False, max_seq_length=4096)
FastVisionModel.for_inference(model)

# --- held-out: seed khác seed train -> mẫu model gần như chưa thấy ---
from datasets import load_dataset
ds = load_dataset("parquet", data_files={"t": args.data}, split="t", streaming=True)
ds = ds.shuffle(seed=args.seed, buffer_size=10000)

passed, i, bad = 0, -1, []
for i, ex in enumerate(ds):
    if i >= args.n: break
    img = to_pil(ex["image"])
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": INSTRUCTION},
    ]}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    code = strip_code(gen)
    ok, err = runs_ok(code)
    passed += ok
    print(f"[{i+1}/{args.n}] {'OK ' if ok else 'ERR'}  pass_rate={passed/(i+1):.1%}")
    if not ok and args.save_bad:
        bad.append({"idx": i, "code": code, "error": err})

n = i + 1
print(f"\n=== EXECUTION PASS RATE: {passed}/{n} = {passed/n:.1%} ===")
if args.save_bad and bad:
    with open(args.save_bad, "w", encoding="utf-8") as f:
        for b in bad: f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"Đã lưu {len(bad)} code lỗi vào {args.save_bad}")
