# Code to Docstring Generation using LLMs
---

## 🔍 Project Overview
This project focuses on **automatically generating Python docstrings** from code using **Large Language Models (LLMs)**.  
We fine-tune the **Qwen2.5** model with **LoRA / QLoRA** techniques and integrate **knowledge graphs** to improve factual accuracy and contextual awareness.

---

## ✨ Features
- Fine-tuning with LoRA / QLoRA adapters  
- Uses CodeSearchNet (Python subset) dataset  
- Knowledge graph integration for context-aware summaries  
- Evaluation with BLEU, ROUGE, and human judgment  
- Easy inference script to generate docstrings

---

## 🧰 Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch transformers accelerate peft datasets sentencepiece bitsandbytes

.
├── README.md
├── data/
│   └── codesearchnet/
├── src/
│   ├── train.py
│   ├── finetune_lora.py
│   ├── qlora_train.py
│   ├── inference.py
│   └── utils.py
├── eval/
│   └── evaluate.py
├── notebooks/
└── LICENSE


