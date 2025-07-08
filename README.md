# Multi‑Stage Domain Adaptive LLM Fine Tuning and Benchmark Evaluation 🦙🚀

This repository provides a **complete, minimal‐but‑extensible pipeline** for fine‑tuning *multiple* open‑source LLMs across several *optimization strategies* (SFT, QLoRA, DPO, GRPO) and *domains* (medical, code‑generation, legal/policy).

## Features

- **Role-based & multi-turn prompt templates**  
  Easily swap personas (doctor, lawyer, coder) and demo a back-and-forth conversation.

- **Chain-of-Thought fine-tuning** (`--cot`)  
  Teach your model to think out loud with CoT examples.

- **Memory-efficient training** via **Unsloth** + Flash-Attention-2  
  Fine-tune 7B–13B models on a single 24 GB GPU with 4-bit quantization.

- **GGUF export** for `ollama`, `llama.cpp`, or `vLLM`  
  Bundle base weights + LoRA adapters into one portable `.gguf` file for ultra-fast inference.

---

## ✨ Quickstart

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Prepare dataset (already generated 50 dummy rows)
python scripts/prepare_dataset.py

# 3. Train (example: QLoRA on Llama‑3‑8B)
python scripts/train.py \
    --models unsloth/llama-3-8b-bnb-4bit \
    --strategy qlora \
    --dataset_jsonl data/dummy_instructions.jsonl \
    --output_dir outputs/checkpoints/llama3-qlora \
    --lora_rank 64 \
    --batch_size 2 \
    --persona doctor \
    --cot --multi_turn
# If you want to train multiple models in one go, just list them:
python scripts/train.py \
    --models unsloth/llama-3-8b-bnb-4bit unsloth/mistral-7b-bnb-4bit \
    --strategy qlora \
    --dataset_jsonl data/dummy_instructions.jsonl \
    --output_dir outputs/multi_model_runs \
    --lora_rank 64 \
    --batch_size 2 \
    --persona doctor \
    --cot --multi_turn

# 4. Evaluate
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/llama3-sft/checkpoint-700 \
  --dataset data/processed.jsonl \
  --persona doctor

# 5. Export adapter → GGUF
python scripts/export_gguf.py \
    --checkpoint outputs/checkpoints/llama3-qlora
```

See **`scripts/train.py`** for full CLI (supports `--strategy {sft,dpo,grpo,qlora}`, `--cot`, `--persona doctor|lawyer|coder`, `--multi_turn`).

---

## 🗂️ Structure

```
llm_finetuning_benchmark/
├── data/                      # JSONL datasets
├── scripts/                   # Training / eval / export
├── utils/                     # Prompt templates, helpers
├── config/                    # YAML config examples
└── outputs/                   # Checkpoints & logs
```

---

## 🛠️ Requirements

* Python ≥ 3.10  
* CUDA ≥ 12, GPU ≥ 24 GB (for 7B/8B models)  
* [Unsloth](https://github.com/unslothai/unsloth) `pip install unsloth`  

Full list in **`requirements.txt`**.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) [![GitHub stars](https://img.shields.io/github/stars/eliashossain001/Domain-adaptive-llm-ft?style=social)](https://github.com/eliashossain001/Domain-adaptive-llm-ft)

## 👨‍💼 Author

**Elias Hossain**  
_Machine Learning Researcher | PhD in Progress | AI x Reasoning Enthusiast_

[![GitHub](https://img.shields.io/badge/GitHub-EliasHossain001-blue?logo=github)](https://github.com/EliasHossain001)
