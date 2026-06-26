"""Đóng gói MathCoderVLM -> ONNX int8 cho CPU (chạy 1 lần, máy có checkpoint).

GGUF/llama.cpp KHÔNG dùng được: MathCoderVLM ghép tay (InternViT+projector+Qwen),
ko engine nào convert tự động. -> đi ONNX, ta tự trace nên kiến trúc gì cũng xuất được.
(package_model.py là cho Qwen3-VL-8B chuẩn HF, khác model này.)

Xuất 2 phần:
  1. encoder+projector  -> vision.onnx   (forward thuần, export tay)
  2. decoder (Qwen+LoRA) -> decoder ONNX  (qua optimum, có KV-cache + lo generate)
Rồi quantize động int8 cả hai (QInt8 trên Linear/MatMul -> ~4x nhỏ, CPU nhanh hơn).

Chạy: python export_onnx.py --ckpt out/stage2/final --out pack_cpu
Giao cả thư mục pack_cpu/ + infer_cpu.py cho user.
Cài: pip install optimum[onnxruntime] onnx onnxruntime
"""
import argparse, json, shutil
from pathlib import Path
import torch
from model import MathCoderVLM, IMAGE_TOKEN
from data import N_VIS, IMG_SIZE, INSTRUCTION

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out/stage2/final")
ap.add_argument("--out", default="pack_cpu")
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

# fp32: export + int8 quant chạy trên CPU, fp32 ổn định nhất (bf16 export hay vướng op chưa support).
print("Load model (fp32)...")
m = MathCoderVLM.from_pretrained(args.ckpt, dtype=torch.float32).eval()
m.decoder = m.decoder.merge_and_unload()   # gộp LoRA -> decoder thành Qwen thường, export được


# --- 1. encoder + projector -> vision.onnx ---------------------------------
class Vision(torch.nn.Module):
    """pixel_values [B,3,H,W] -> vision embeds [B,N_VIS,d_llm]. Bọc encode_images để export."""
    def __init__(self, mm): super().__init__(); self.mm = mm
    def forward(self, pixel_values): return self.mm.encode_images(pixel_values)

vision = Vision(m).eval()
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
vpath = out / "vision_fp32.onnx"
print("Export vision.onnx...")
with torch.no_grad():
    torch.onnx.export(
        vision, dummy, str(vpath),
        input_names=["pixel_values"], output_names=["vision_embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "vision_embeds": {0: "batch"}},
        opset_version=17,
    )

# --- 2. decoder -> ONNX qua optimum (KV-cache, generate-ready) --------------
print("Export decoder ONNX (optimum)...")
dec_dir = out / "decoder_onnx"
tmp = out / "_decoder_hf"; m.decoder.save_pretrained(tmp); m.tok.save_pretrained(tmp)
from optimum.onnxruntime import ORTModelForCausalLM
ORTModelForCausalLM.from_pretrained(tmp, export=True).save_pretrained(dec_dir)
shutil.rmtree(tmp)

# --- 3. quantize động int8 (cả hai) ----------------------------------------
print("Quantize int8...")
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(str(vpath), str(out / "vision_int8.onnx"), weight_type=QuantType.QInt8)
vpath.unlink()
for f in dec_dir.glob("*.onnx"):                 # model.onnx (+ decoder_with_past nếu có)
    quantize_dynamic(str(f), str(f), weight_type=QuantType.QInt8)

# --- 4. tokenizer + meta ----------------------------------------------------
m.tok.save_pretrained(out / "tokenizer")
(out / "meta.json").write_text(json.dumps({
    "n_vis": N_VIS, "img_size": IMG_SIZE, "image_token": IMAGE_TOKEN,
    "image_token_id": m.image_token_id, "instruction": INSTRUCTION,
}, indent=2))
print(f"Done -> {out}/  (vision_int8.onnx, decoder_onnx/, tokenizer/, meta.json)")
