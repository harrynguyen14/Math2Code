"""Eval stage1 (projector align, decoder freeze). KO đo exec pass-rate — stage1 chưa
train sinh code. Đo: eval loss trên held-out + in vài code generate để mắt thường.

Chạy: python eval_stage1.py --n 64 --show 3
"""
import argparse, glob, torch
import pyarrow.parquet as pq
from model import MathCoderVLM
from data import Collator, INSTRUCTION, N_VIS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out/stage1")
ap.add_argument("--shards", default="/workspace/data/math-dataset/Python_rg/*.parquet")
ap.add_argument("--n", type=int, default=64, help="số mẫu held-out đo loss")
ap.add_argument("--show", type=int, default=3, help="số sample in code generate")
ap.add_argument("--max-new", type=int, default=512)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = MathCoderVLM().to(dev)
m.projector.load_state_dict(torch.load(f"{args.ckpt}/projector.pt", map_location=dev))
m.eval()
col = Collator(m)

# held-out: lấy cuối shard đầu (train streaming đọc từ đầu -> cuối ít bị thấy)
shard = sorted(glob.glob(args.shards))[0]
tbl = pq.read_table(shard, columns=["image", "text"])
rows = tbl.slice(max(0, tbl.num_rows - args.n), args.n).to_pylist()

# --- eval loss ---
losses = []
with torch.no_grad():
    for i in range(0, len(rows), 4):
        batch = col(rows[i:i + 4])
        batch = {k: v.to(dev) for k, v in batch.items()}
        losses.append(m(**batch).loss.item())
print(f"eval loss ({len(rows)} mẫu): {sum(losses)/len(losses):.4f}")

# --- generate vài sample để mắt thường (code có ra đúng dạng Python ko) ---
prompt = (f"<|im_start|>user\n{'<|image_pad|>' * N_VIS}{INSTRUCTION}"
          f"<|im_end|>\n<|im_start|>assistant\n")
for ex in rows[:args.show]:
    px = col([ex])["pixel_values"].to(dev)
    ids = m.tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    embeds = m.decoder.get_input_embeddings()(ids)
    vis = m.encode_images(px).to(embeds.dtype)
    embeds[ids == m.image_token_id] = vis.reshape(-1, vis.shape[-1])
    with torch.no_grad():
        out = m.decoder.generate(inputs_embeds=embeds, max_new_tokens=args.max_new, do_sample=False)
    print("\n--- GEN ---\n" + m.tok.decode(out[0], skip_special_tokens=True)[:600])
    print("--- GOLD ---\n" + ex["text"][:300])
