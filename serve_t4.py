"""Serve Qwen3-VL math specialist trên 1 GPU T4 15GB (load 4-bit bitsandbytes).
Adapter LoRA áp lên base lúc load (QLoRA inference) -> không cần merge.

Cài:  pip install vllm  (vllm tự kéo transformers/bitsandbytes phù hợp)
Chạy: python serve_t4.py
Gọi:  OpenAI-compatible API tại http://localhost:8000/v1 , model name = "math"
"""
import argparse, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="unsloth/Qwen3-VL-8B-Instruct")
ap.add_argument("--adapter", default="harryrobert/Qwen-3-VL-Math-Spec")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--max-len", type=int, default=8192, help="ctx len; giảm nếu OOM trên T4")
args = ap.parse_args()

# vLLM serve: 4-bit bitsandbytes để 8B vừa T4 15GB, bật LoRA adapter
cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", args.base,
    "--quantization", "bitsandbytes",
    "--load-format", "bitsandbytes",
    "--enable-lora",
    "--lora-modules", f"math={args.adapter}",
    "--max-model-len", str(args.max_len),
    "--gpu-memory-utilization", "0.92",
    "--port", str(args.port),
    "--dtype", "float16",            # T4 không có bf16 tensor core
    "--max-num-seqs", "4",           # tải thấp; tăng nếu cần throughput
]
print("Launching vLLM:\n  " + " ".join(cmd) + "\n")
print(f"Khi sẵn sàng, gọi model name = 'math' tại http://localhost:{args.port}/v1\n")
subprocess.run(cmd, check=True)
