"""Eval stage2: exec pass-rate phân tầng theo shard (stage2 đã train sinh code).
Load MathCoderVLM local (out/stage2/final), generate code mỗi ảnh, chạy thử (matplotlib Agg).

Chạy: python eval_stage2.py --per-type 20
"""
import argparse, glob, os, re, subprocess, sys, tempfile, torch
import pyarrow.parquet as pq
from tqdm import tqdm
from model import MathCoderVLM
from data import Collator, INSTRUCTION, N_VIS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out/stage2/final")
ap.add_argument("--shards", default="/workspace/data/math-dataset/Python_rg/*.parquet")
ap.add_argument("--per-type", type=int, default=20, help="số mẫu mỗi shard")
ap.add_argument("--max-new", type=int, default=3072)
ap.add_argument("--timeout", type=int, default=30)
ap.add_argument("--batch", type=int, default=8, help="số ảnh / lần generate")
args = ap.parse_args()

PREFIX = ("import matplotlib\nmatplotlib.use('Agg')\n"
          "import matplotlib.pyplot as _p\n_p.show=lambda *a,**k:None\n")

def strip_code(t):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", t, re.DOTALL)
    if m: return m.group(1).strip()
    for i, l in enumerate(t.splitlines()):
        if l.lstrip().startswith(("import ", "from ")):
            return "\n".join(t.splitlines()[i:]).strip()
    return t.strip()

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

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = MathCoderVLM.from_pretrained(args.ckpt).to(dev)
m.eval()
col = Collator(m)
prompt = (f"<|im_start|>user\n{'<|image_pad|>' * N_VIS}{INSTRUCTION}"
          f"<|im_end|>\n<|im_start|>assistant\n")
ids = m.tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)  # [1,T]
eos = m.tok.eos_token_id
GEN = dict(do_sample=False, temperature=None, top_p=None, top_k=None,  # greedy sạch -> hết warning
           use_cache=True, pad_token_id=eos, eos_token_id=eos)

def generate_batch(px):  # px: [B,3,H,W]. Prompt giống hệt mọi mẫu -> embed tile B lần, ko cần pad.
    B = px.shape[0]
    ids_b = ids.expand(B, -1)                                  # [B,T] cùng prompt
    embeds = m.decoder.get_input_embeddings()(ids_b)           # [B,T,d]
    px = px.to(next(m.encoder.parameters()).dtype)             # encoder bf16; eval ko autocast
    vis = m.encode_images(px).to(embeds.dtype)                 # [B,N_VIS,d]
    mask = (ids_b == m.image_token_id)                         # [B,T]; image token cùng vị trí mọi hàng
    embeds[mask] = vis.reshape(-1, vis.shape[-1])              # row-major -> khớp thứ tự (mẫu i, token j)
    with torch.no_grad():
        out = m.decoder.generate(inputs_embeds=embeds, attention_mask=torch.ones_like(ids_b),
                                 max_new_tokens=args.max_new, **GEN)
    return m.tok.batch_decode(out, skip_special_tokens=True)   # [B] strings

shards = sorted(glob.glob(args.shards))
print(f"Found {len(shards)} shards\n")
overall_pass, overall_n, results = 0, 0, {}
for shard in shards:
    name = os.path.basename(shard)
    tbl = pq.read_table(shard, columns=["image", "text"])
    take = min(args.per_type, tbl.num_rows)
    rows = tbl.slice(max(0, tbl.num_rows - take), take).to_pylist()  # held-out cuối shard
    passed = 0
    for i in tqdm(range(0, take, args.batch), desc=name, leave=False):
        chunk = rows[i:i + args.batch]
        px = col(chunk)["pixel_values"].to(dev)
        for raw in generate_batch(px):
            passed += runs_ok(strip_code(raw))
    rate = passed / take
    results[name] = (passed, take, rate)
    overall_pass += passed; overall_n += take
    print(f"{name:35s} {passed:3d}/{take:3d} = {rate:.0%}")

print("\n" + "=" * 50)
print(f"OVERALL EXEC PASS RATE: {overall_pass}/{overall_n} = {overall_pass/overall_n:.1%}")
print("=" * 50)
weak = [n for n, (_, _, r) in results.items() if r < 0.8]
if weak:
    print("\nLOAI YEU (<80%):")
    for n in weak: print(" -", n, f"{results[n][2]:.0%}")
