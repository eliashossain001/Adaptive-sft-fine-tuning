#!/usr/bin/env python
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Export a LoRA-fine-tuned checkpoint to GGUF via llama.cpp"
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model ID or local folder (e.g. unsloth/llama-3-8b-bnb-4bit or /path/to/base-4bit/)"
    )
    parser.add_argument(
        "--adapter-folder",
        required=True,
        help="Folder containing your adapter (adapter_model.safetensors + adapter_config.json)"
    )
    parser.add_argument(
        "--outfile",
        help="Output GGUF path (defaults to <adapter_folder>.gguf)"
    )
    parser.add_argument(
        "--convert-script",
        default="convert-pth-to-gguf.py",
        help="Path to llama.cpp's conversion script"
    )
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_folder)
    if not adapter_dir.is_dir():
        sys.exit(f"[ERROR] Adapter folder not found: {adapter_dir}")

    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.exists():
        sys.exit(f"[ERROR] adapter_model.safetensors missing in {adapter_dir}")

    # Decide output path
    outfile = Path(args.outfile) if args.outfile else adapter_dir.with_suffix(".gguf")

    # Build conversion command
    cmd = [
        sys.executable,
        args.convert_script,
        "--base",             args.base_model,
        "--adapter",          str(adapter_file),
        "--tokenizer",        str(adapter_dir),
        "--outfile",          str(outfile),
    ]

    print(f"[INFO] Running conversion:\n    {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    print(f"[✔] Export complete! GGUF saved to {outfile}")

if __name__ == "__main__":
    main()
