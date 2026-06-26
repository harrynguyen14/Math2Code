"""Chạy MathCoderVLM (đã đóng gói ONNX int8) trên CPU. Giao file này + thư mục pack_cpu/ cho user.

Ảnh -> vision_int8.onnx -> vision embeds -> nối vào decoder ONNX -> greedy generate -> code Python.

Cài: pip install onnxruntime optimum[onnxruntime] transformers pillow
Chạy: python infer_cpu.py anh.png
       python infer_cpu.py anh.png --pack pack_cpu --max-new 1024
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from optimum.onnxruntime import ORTModelForCausalLM
import onnxruntime as ort
import torch  # chỉ để dựng inputs_embeds tensor cho optimum; ko load model torch

VIT = "OpenGVLab/InternViT-300M-448px-V2_5"  # image processor (resize/normalize); ko tải weight

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--pack", default="pack_cpu")
ap.add_argument("--max-new", type=int, default=1024)  # train max_len=1024; >đó model chưa từng sinh
args = ap.parse_args()
pack = Path(args.pack)
meta = json.loads((pack / "meta.json").read_text())

tok = AutoTokenizer.from_pretrained(pack / "tokenizer")
proc = CLIPImageProcessor.from_pretrained(VIT, use_fast=True)
vis_sess = ort.InferenceSession(str(pack / "vision_int8.onnx"), providers=["CPUExecutionProvider"])
dec = ORTModelForCausalLM.from_pretrained(pack / "decoder_onnx", provider="CPUExecutionProvider")

# ảnh -> vision embeds [1, N_VIS, d]
img = Image.open(args.image).convert("RGB")
px = proc(img, return_tensors="np").pixel_values.astype(np.float32)
vis = vis_sess.run(None, {"pixel_values": px})[0]      # [1, N_VIS, d_llm]

# prompt giống lúc train/eval; chèn N_VIS image token, thay embed bằng vision
prompt = (f"<|im_start|>user\n{meta['image_token'] * meta['n_vis']}{meta['instruction']}"
          f"<|im_end|>\n<|im_start|>assistant\n")
ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids   # [1,T]
embeds = dec.get_input_embeddings()(ids)                                     # [1,T,d]
mask = ids == meta["image_token_id"]
embeds[mask] = torch.from_numpy(vis).reshape(-1, vis.shape[-1]).to(embeds.dtype)

out = dec.generate(inputs_embeds=embeds, attention_mask=torch.ones_like(ids),
                   max_new_tokens=args.max_new, do_sample=False,
                   pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True))
