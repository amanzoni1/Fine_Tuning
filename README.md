# Fine-Tuning Portfolio

This repository showcases a collection of Colab notebooks exploring different fine-tuning strategies for large language models, all runnable on a free Colab T4 GPU. The notebooks cover a range of techniques from supervised fine-tuning (SFT) to reinforcement learning (RL), demonstrating various approaches to training and optimizing LLMs for specific tasks.
Each notebook emphasizes practical efficiency methods to keep memory usage in check, including 4-bit quantization with bitsandbytes, LoRA adapters for parameter-efficient fine-tuning (PEFT), gradient checkpointing for memory optimization, fast batch inference etc. These techniques make it possible to train large models on consumer hardware while maintaining competitive performance.

## Notebooks

### 1. Reinforcement Learning - LLama3.1-8B with GRPO & CoT Reasoning
- **Filename:** `RL_LLama3_1_8B_GRPO.ipynb`
- **Colab Link:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amanzoni1/fine_tuning/blob/main/RL_LLama3_1_8B_GRPO.ipynb)
- **Description:**  
Two-phase training approach that transforms LLaMA 3.1-8B into a reasoning-capable text summarization model using Unsloth for significant performance improvements. Begins with supervised fine-tuning on GSM8K mathematical problems to teach chain-of-thought reasoning format, then applies Group Relative Policy Optimization (GRPO) on XSum summarization dataset. Uses custom reward functions evaluating format compliance and content quality through ROUGE scores, BERTScore semantic similarity, and length penalties. Combines 4-bit quantization with LoRA adapters for efficient VRAM usage while leveraging Unsloth's optimizations for faster training and inference.

### 2. Fine-Tuning Gemma-7B-It for Structured Output & Financial Sentiment Classification
- **Filename:** `SFT_Gemma_7B_it.ipynb`  
- **Colab Link:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amanzoni1/fine_tuning/blob/main/SFT_Gemma_7B_it.ipynb)
- **Description:**  
Transforms Gemma-7B-Instruct into a domain-specialized financial sentiment analyst that outputs clean JSON objects. Provides an in-depth exploration of dataset characteristics and tokenizer behavior, diving deep into sentence length distributions, vocabulary analysis, and tokenization patterns to understand the fundamental building blocks of the training process. Uses the Financial PhraseBank dataset with comprehensive data preprocessing and applies QLoRA (4-bit quantization + LoRA) with activation checkpointing for memory efficiency. Includes thorough evaluation with token-level metrics, accuracy, precision, recall, and F1 scores, plus analysis of temperature effects on JSON output quality.

---

> _Feel free to browse each notebook for detailed code, hyperparameters, and results.  Contributions and improvements are welcome!_
