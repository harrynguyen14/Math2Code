import argparse, subprocess, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252 không in được tiếng Việt

ADAPTER = r"D:\math-dataset\ckp\qwen3_vl_math"
BASE = "unsloth/Qwen3-VL-8B-Instruct"
OUT = Path("out")
MERGED = OUT / "merged"
GGUF = OUT / "gguf"


def merge():
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    MERGED.mkdir(parents=True, exist_ok=True)
    print(f"Load base {BASE} (fp16)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True,  # ponytail: tránh nhân đôi RAM lúc load -> đỡ lag
    )
    print(f"Áp adapter {ADAPTER}...")
    model = PeftModel.from_pretrained(model, ADAPTER)
    print("Merge...")
    model = model.merge_and_unload()
    model.save_pretrained(MERGED, safe_serialization=True)
    AutoProcessor.from_pretrained(ADAPTER).save_pretrained(MERGED)  # tokenizer + image processor
    print(f"OK -> {MERGED}")


# source tree đúng tag b9763 (convert_hf_to_gguf.py mới import cả package -> cần nguyên cây)
LLAMA_SRC = Path(r"D:\llama-src")
# winget đặt .exe ở đây; sửa nếu bạn cài chỗ khác
WINGET_BIN = Path.home() / r"AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe"


def _find_quantize(llama_cpp):
    # ưu tiên --llama-cpp; rồi winget; rồi PATH
    cands = []
    if llama_cpp:
        lc = Path(llama_cpp)
        cands += [lc / "build" / "bin" / "llama-quantize.exe", lc / "llama-quantize.exe"]
    cands.append(WINGET_BIN / "llama-quantize.exe")
    for c in cands:
        if c.exists():
            return str(c)
    return "llama-quantize"  # giả định đã trên PATH (mở shell mới sau winget)


def gguf(llama_cpp: str):
    assert MERGED.exists(), "Chạy `merge` trước"
    GGUF.mkdir(parents=True, exist_ok=True)
    f16 = GGUF / "qwen3vl-math-f16.gguf"
    q4 = GGUF / "qwen3vl-math-Q4_K_M.gguf"
    mmproj = GGUF / "mmproj-qwen3vl-math-f16.gguf"

    conv = LLAMA_SRC / "convert_hf_to_gguf.py"
    assert conv.exists(), f"Không thấy {conv} — tải source llama.cpp tag b9763 về {LLAMA_SRC}"

    # LM -> f16 gguf (KHÔNG --mmproj: cờ đó chỉ xuất riêng vision encoder)
    print("Convert HF -> GGUF (f16) [language model]...")
    subprocess.run([sys.executable, str(conv), str(MERGED),
                    "--outfile", str(f16), "--outtype", "f16"], check=True)

    # vision encoder -> file mmproj riêng (giữ f16, không quantize)
    print("Convert vision encoder -> mmproj...")
    subprocess.run([sys.executable, str(conv), str(MERGED),
                    "--outfile", str(mmproj), "--outtype", "f16",
                    "--mmproj"], check=True)

    # quantize LM xuống Q4_K_M
    quant = _find_quantize(llama_cpp)
    print(f"Quantize -> Q4_K_M  (dùng {quant})...")
    subprocess.run([quant, str(f16), str(q4), "Q4_K_M"], check=True)
    print(f"OK -> {q4}\n     mmproj -> {mmproj}\nChạy: llama-server -m {q4} --mmproj {mmproj}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["merge", "gguf", "all"])
    ap.add_argument("--llama-cpp", help="thư mục llama.cpp (cho gguf)")
    a = ap.parse_args()
    if a.step in ("merge", "all"):
        merge()
    if a.step in ("gguf", "all"):
        gguf(a.llama_cpp)
