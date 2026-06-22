"""Push adapter LoRA lên HuggingFace Hub.
Chạy: python push_model.py --token hf_xxx
      python push_model.py --token hf_xxx --public   # nếu muốn public
"""
import argparse
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--token", required=True, help="HF token có quyền WRITE")
ap.add_argument("--repo", default="harryrobert/Qwen-3-VL-Math-Spec")
ap.add_argument("--folder", default=r"D:\math-dataset\ckp\qwen3_vl_math", help="thư mục adapter")
ap.add_argument("--public", action="store_true", help="tạo repo public (mặc định private)")
args = ap.parse_args()

api = HfApi(token=args.token)
api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=args.folder)
print(f"Done -> https://huggingface.co/{args.repo}")
