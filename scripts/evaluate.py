#!/usr/bin/env python
# scripts/evaluate.py

"""
Evaluate a LoRA-fine-tuned Unsloth model (or any HF model+adapter) on a local JSONL dataset.
This version ensures it imports the real Hugging Face `evaluate` library rather than the local script.
"""
from pathlib import Path
import sys

# ─── 0. Ensure 'utils.prompts' is importable by adding project root to PYTHONPATH
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── 1. Temporarily remove the script directory from sys.path so `import evaluate` grabs the HF lib
if str(THIS_DIR) in sys.path:
    sys.path.remove(str(THIS_DIR))
import evaluate as hf_evaluate
# ─── 2. Restore the script directory so we can import `utils.prompts`
sys.path.insert(0, str(THIS_DIR))

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils.prompts import build_prompt


def find_adapter_dir(path: Path) -> Path:
    """
    Locate the directory containing adapter_config.json under the given path.
    """
    p = Path(path)
    if (p / "adapter_config.json").exists():
        return p
    for sub in sorted(p.iterdir()):
        if sub.is_dir() and (sub / "adapter_config.json").exists():
            return sub
    raise FileNotFoundError(f"No adapter_config.json found under {p}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a LoRA-fine-tuned Unsloth model checkpoint"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to adapter folder or its parent (e.g. outputs/.../llama3-qlora or checkpoint-700)"
    )
    parser.add_argument(
        "--dataset", default="data/processed.jsonl",
        help="Local JSONL file with fields: instruction, input, output"
    )
    parser.add_argument(
        "--persona", default="doctor",
        choices=["doctor", "lawyer", "coder"],
        help="Persona template to use"
    )
    args = parser.parse_args()

    root = Path(args.checkpoint)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {root}")

    # 3. Locate adapter directory containing adapter_config.json
    adapter_dir = find_adapter_dir(root)

    # 4. Read base model name from adapter_config.json
    cfg = json.load(open(adapter_dir / "adapter_config.json"))
    base_name = cfg.get("base_model_name_or_path")
    if not base_name:
        raise KeyError("adapter_config.json missing 'base_model_name_or_path'")

    # 5. Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # 6. Attach LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir.as_posix(),
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 7. Load tokenizer from adapter folder
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir.as_posix())

    # 8. Prepare BLEU metric
    bleu = hf_evaluate.load("bleu")
    refs, preds = [], []

    # 9. Iterate dataset, generate, collect
    with open(args.dataset, "r") as f:
        for line in f:
            ex = json.loads(line)
            prompt = build_prompt(args.persona, ex["instruction"], ex["input"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=128)
            pred = tokenizer.decode(out[0], skip_special_tokens=True)
            refs.append([ex["output"]])
            preds.append(pred)

    # 10. Compute BLEU
    result = bleu.compute(predictions=preds, references=refs)
    print(f"BLEU score: {result['bleu']:.4f}")


if __name__ == "__main__":
    main()
