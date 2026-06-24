"""Eval stage2: exec pass-rate phân tầng theo shard (stage2 đã train sinh code).
Load MathCoderVLM local (out/stage2/final), generate code mỗi ảnh, chạy thử (matplotlib Agg).

Chạy: python eval_stage2.py --per-type 20
"""
import argparse, glob, os, re, subprocess, sys, tempfile, torch
import pyarrow.parquet as pq
from model import MathCoderVLM
from data import Collator, INSTRUCTION, N_VIS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out/stage2/final")
ap.add_argument("--shards", default="/workspace/data/math-dataset/Python_rg/*.parquet")
ap.add_argument("--per-type", type=int, default=20, help="số mẫu mỗi shard")
ap.add_argument("--max-new", type=int, default=3072)
ap.add_argument("--timeout", type=int, default=30)
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
ids = m.tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)

def generate(px):  # chèn vision embed vào prompt rồi generate (batch=1)
    embeds = m.decoder.get_input_embeddings()(ids)
    vis = m.encode_images(px).to(embeds.dtype)
    embeds[ids == m.image_token_id] = vis.reshape(-1, vis.shape[-1])
    with torch.no_grad():
        out = m.decoder.generate(inputs_embeds=embeds, max_new_tokens=args.max_new, do_sample=False)
    return m.tok.decode(out[0], skip_special_tokens=True)

shards = sorted(glob.glob(args.shards))
print(f"Found {len(shards)} shards\n")
overall_pass, overall_n, results = 0, 0, {}
for shard in shards:
    name = os.path.basename(shard)
    tbl = pq.read_table(shard, columns=["image", "text"])
    take = min(args.per_type, tbl.num_rows)
    rows = tbl.slice(max(0, tbl.num_rows - take), take).to_pylist()  # held-out cuối shard
    passed = 0
    for ex in rows:
        px = col([ex])["pixel_values"].to(dev)
        passed += runs_ok(strip_code(generate(px)))
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
