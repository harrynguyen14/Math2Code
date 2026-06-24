"""VLM lệch: InternViT-300M (encoder) + MLP projector + Qwen2.5-Coder-1.5B (decoder).

Tự ghép nn.Module mỏng thay vì ép vào LlavaForConditionalGeneration:
InternViT là custom code (trust_remote_code), interface không khớp Llava class -> ghép tay
để kiểm soát luồng + dễ debug.

Luồng: ảnh -> InternViT -> [B, 1024, 1024] -> pixel_shuffle (gộp 2x2)
       -> [B, 256, 4096] -> MLP -> [B, 256, 1536]
       -> chèn vào chỗ <image> trong embeds của Coder -> decoder -> code.

pixel_shuffle giảm 4x vision token (1024->256): cắt 4x sequence length của decoder
-> ~4x nhanh attention. Đây là tối ưu lớn nhất, lấy từ InternVL gốc.

self-check: python model.py  (forward 1 ảnh giả, kiểm tra shape + không NaN)
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor

VIT = "OpenGVLab/InternViT-300M-448px-V2_5"
LLM = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
IMAGE_TOKEN = "<|image_pad|>"  # token giữ chỗ cho vision tokens trong chuỗi text


class MathCoderVLM(nn.Module):
    def __init__(self, vit=VIT, llm=LLM, dtype=torch.bfloat16):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(vit, trust_remote_code=True, torch_dtype=dtype)
        self.decoder = AutoModelForCausalLM.from_pretrained(
            llm, torch_dtype=dtype, attn_implementation="flash_attention_2")
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2
            apply_liger_kernel_to_qwen2(model=self.decoder)
        except ImportError:
            print("liger_kernel chưa cài (pip install liger-kernel) -> dùng kernel mặc định")
        self.tok = AutoTokenizer.from_pretrained(llm)
        if IMAGE_TOKEN not in self.tok.get_vocab():
            self.tok.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
            self.decoder.resize_token_embeddings(len(self.tok))
        self.image_token_id = self.tok.convert_tokens_to_ids(IMAGE_TOKEN)

        self.downsample = 0.5                          # 2x2 -> 1 token: 1024->256
        d_vit = self.encoder.config.hidden_size        # 1024
        d_llm = self.decoder.config.hidden_size        # 1536
        d_shuffled = int(d_vit / (self.downsample ** 2))  # 1024 * 4 = 4096
        self.projector = nn.Sequential(
            nn.Linear(d_shuffled, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm),
        ).to(dtype)
        # use_fast: image processor Rust-backed, nhanh hơn nhiều bản PIL thuần
        self.image_processor = CLIPImageProcessor.from_pretrained(vit, use_fast=True)

    def pixel_shuffle(self, x):
        # x: [B, H*W, C] vuông -> gộp khối 2x2 patch thành 1 token, C *= 4.
        # Y hệt InternVL gốc: space-to-depth trên lưới patch.
        B, N, C = x.shape
        h = w = int(N ** 0.5)
        x = x.reshape(B, h, w, C)
        s = self.downsample
        x = x.reshape(B, h, int(w * s), int(C / s))                 # gộp theo W
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(B, int(w * s), int(h * s), int(C / (s * s)))  # gộp theo H
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.reshape(B, int(N * s * s), int(C / (s * s)))       # [B, N/4, C*4]

    def encode_images(self, pixel_values):
        # Encoder LUÔN freeze (cả 2 stage) -> no_grad: bỏ activation + backward của
        # InternViT (phần nặng nhất). Chỉ projector giữ grad. Đây là tốc-độ kiểu Unsloth:
        # không backward qua phần đóng băng.
        with torch.no_grad():
            out = self.encoder(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
            out = self.pixel_shuffle(out)              # [B, n_patch/4, 4096]
        return self.projector(out)  # [B, n_patch/4, d_llm] — grad chỉ từ đây

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        embeds = self.decoder.get_input_embeddings()(input_ids)  # [B, T, d_llm]
        if pixel_values is not None:
            vis = self.encode_images(pixel_values).to(embeds.dtype)  # [B, n_vis, d_llm]
            # thay embed tại vị trí image token bằng vision embeds (theo thứ tự xuất hiện)
            mask = input_ids == self.image_token_id
            embeds[mask] = vis.reshape(-1, vis.shape[-1])
        return self.decoder(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)

    def set_trainable(self, projector=True, decoder_lora=False, encoder=False):
        # đặt requires_grad theo giai đoạn; LoRA gắn ngoài (peft) nên ở đây chỉ bật/tắt thô
        for p in self.encoder.parameters(): p.requires_grad_(encoder)
        for p in self.projector.parameters(): p.requires_grad_(projector)
        for p in self.decoder.parameters(): p.requires_grad_(decoder_lora)

    def save_pretrained(self, out_dir):
        """Lưu 1 thư mục tự-chứa load lại được: decoder (LoRA adapter hoặc full),
        projector, tokenizer. Encoder freeze -> không lưu, from_pretrained tải lại từ hub.
        Đây là format HF chuẩn cho phần train được; deploy engine (vLLM) tính sau."""
        from pathlib import Path
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        self.decoder.save_pretrained(out / "decoder")   # PEFT lưu adapter, model thường lưu full
        torch.save(self.projector.state_dict(), out / "projector.pt")
        self.tok.save_pretrained(out / "tokenizer")     # gồm cả IMAGE_TOKEN đã thêm

    @classmethod
    def from_pretrained(cls, out_dir, vit=VIT, llm=LLM, dtype=torch.bfloat16):
        """Dựng lại model: base (encoder từ hub + Qwen base) rồi nạp decoder/projector/tok đã train."""
        from pathlib import Path
        from peft import PeftModel
        out = Path(out_dir)
        m = cls(vit=vit, llm=llm, dtype=dtype)
        m.tok = AutoTokenizer.from_pretrained(out / "tokenizer")
        m.decoder.resize_token_embeddings(len(m.tok))
        m.image_token_id = m.tok.convert_tokens_to_ids(IMAGE_TOKEN)
        dec = out / "decoder"
        if (dec / "adapter_config.json").exists():       # stage2: LoRA adapter
            m.decoder = PeftModel.from_pretrained(m.decoder, dec)
        else:                                            # full decoder
            m.decoder = AutoModelForCausalLM.from_pretrained(dec, torch_dtype=dtype)
        m.projector.load_state_dict(torch.load(out / "projector.pt", map_location="cpu"))
        return m


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = MathCoderVLM().to(dev)         # InternViT flash_attn cần CUDA
    n_patch = (448 // 14) ** 2          # 1024 patches @448px
    n_vis = n_patch // 4               # sau pixel_shuffle: 256 token
    px = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16, device=dev)
    # dựng input: prompt có n_vis image token
    img_tokens = IMAGE_TOKEN * n_vis
    text = f"<|im_start|>user\n{img_tokens}Write the Python code.<|im_end|>\n<|im_start|>assistant\n"
    enc = m.tok(text, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = m(enc.input_ids, enc.attention_mask, px)
    logits = out.logits
    print("logits shape:", logits.shape, "| n_vis:", n_vis)
    assert not torch.isnan(logits).any(), "NaN trong logits!"
    # Qwen2.5 đệm embedding (151936) > len(tok) (~151665) cho hiệu năng -> so với head, ko phải tok
    assert logits.shape[-1] == m.decoder.get_output_embeddings().weight.shape[0], "vocab size lệch"
    print("OK: forward chạy, không NaN, vocab khớp.")
