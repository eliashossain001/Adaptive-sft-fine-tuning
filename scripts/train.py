#!/usr/bin/env python
# scripts/train.py

import argparse
import json
import nltk
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# Download punkt for any tokenization (optional)
nltk.download("punkt")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune an Unsⱽloth LLM on a JSONL instruction-input-output dataset"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Unsloth model name (e.g. unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit)",
    )
    parser.add_argument(
        "--dataset_jsonl",
        required=True,
        help="Path to local JSONL with fields: instruction, input, output",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Where to save checkpoints & tokenizer"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the base model in 4-bit (requires Unsⱽloth-bnb checkpoint)",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=64, help="LoRA rank for QLoRA adapters"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=2500)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 (requires Ampere+ GPU)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model & tokenizer to HF Hub after training",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token (if pushing to hub)",
    )
    args = parser.parse_args()

    # ── 1. Load local JSONL dataset ───────────────────────────────────────────
    raw = load_dataset("json", data_files={"train": args.dataset_jsonl})["train"]
    raw = raw.shuffle(seed=args.seed).train_test_split(test_size=0.15, seed=args.seed)
    train_ds = raw["train"]

    # ── 2. Define Alpaca-style prompt template ────────────────────────────────
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    # ── 3. Map to a single "text" field ────────────────────────────────────────
    def formatting_prompts_func(examples):
        texts = []
        for ins, inp, out in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            txt = alpaca_prompt.format(ins, inp, out) + tokenizer.eos_token
            texts.append(txt)
        return {"text": texts}

    # ── 4. Load model & tokenizer ─────────────────────────────────────────────
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    setattr(model, "tokenizer", tokenizer)  # ensures unsloth.utilities work

    # ── 5. Apply LoRA adapters (QLoRA) ────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_rank,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # ── 6. Prepare training split ──────────────────────────────────────────────
    train_ds = train_ds.map(
        formatting_prompts_func, batched=True, remove_columns=train_ds.column_names
    )

    # ── 7. Configure the TRL SFTTrainer ───────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=4096,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=False,
            bf16=args.bf16,
            logging_steps=50,
            save_steps=700,
            optim="adamw_8bit",
            max_grad_norm=1.0,
            weight_decay=0.05,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=args.output_dir,
            report_to="none",
        ),
    )

    # ── 8. Train! ──────────────────────────────────────────────────────────────
    trainer.train()

    # ── 9. Save & (optionally) push to Hub ────────────────────────────────────
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if args.push_to_hub:
        model.push_to_hub(
            repo_name=args.output_dir.split("/")[-1],
            private=True,
            use_auth_token=args.hf_token,
        )
        tokenizer.push_to_hub(
            repo_name=args.output_dir.split("/")[-1],
            private=True,
            use_auth_token=args.hf_token,
        )


if __name__ == "__main__":
    main()
