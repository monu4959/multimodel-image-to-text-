[README.md](https://github.com/user-attachments/files/26861824/README.md)
# Image-to-Text Generation Using CLIP and GPT-2 with LoRA Fine-Tuning

**Course:** IE 7615 – Deep Learning for AI  
**Northeastern University – Department of Data Analytics Engineering**

**Authors:** Arvind Deivanayagam · Rohan Kolla · Sadhvika Reddy Andem · Sai Venkata Kashyap Akula

---

## Overview

This project implements an image captioning system that bridges the **CLIP visual encoder** and the **GPT-2 language model** using a learned MLP projection layer with prefix conditioning and Low-Rank Adaptation (LoRA). Rather than training a unified model from scratch, both pretrained components are kept frozen and only the projection layer and LoRA adapters are updated.

The project follows a systematic experimental progression across four versions (v1–v4), culminating in a **CIDEr score of 0.7668** and **BLEU-4 of 0.2582** on the COCO test set — outperforming the pretrained BLIP model despite using 1,290× less training data.

---

## Results Summary

| Version | Architecture | Pairs | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|---|
| v3 | CLIP ViT-B/32 + MLP + GPT-2 + LoRA r16 | 25k | 0.683 | 0.254 | 0.253 | 0.497 | 0.751 |
| **v4** | CLIP ViT-B/32 + MLP + GPT-2 + LoRA r16 | **100k** | **0.695** | **0.258** | **0.262** | **0.497** | **0.767** |
| Method B | CLIP ViT-B/32 + Cross-Attn + GPT-2 + LoRA r16 | 100k | 0.576 | 0.194 | 0.399 | 0.395 | 0.552 |
| Method B+ | CLIP ViT-L/14 + Gated Cross-Attn + GPT-2 + LoRA r32 | 25k | 0.633 | 0.225 | 0.253 | 0.431 | 0.605 |
| BLIP (baseline) | End-to-end ViT+BERT | 129M | 0.599 | 0.250 | 0.229 | 0.426 | 0.731 |

---

## Project Structure

```
image-captioning/
│
├── image_captioning_v4.ipynb        # Primary model — best results
├── image_captioning_v3.ipynb        # v3 — prefix=40, LoRA r16, 5k images
├── image_captioning_method_b.ipynb  # Method B — cross-attention conditioning
├── blip_comparison.ipynb            # BLIP vs v4 comparison
├── LLAVA.ipynb                      # LLaVA-1.6 qualitative comparison
└── README.md
```

---

## Architecture

### Method A — Prefix Conditioning (v1–v4)

```
Image [224×224]
    → CLIP ViT-B/32 (frozen)
    → 512-d embedding
    → MLP projection (2-layer + Tanh)
    → 40 prefix tokens [40 × 768]
    → concatenate with caption tokens
    → GPT-2 base + LoRA rank 16 (frozen base weights)
    → cross-entropy loss on caption positions only
```

### Method B — Cross-Attention Conditioning

```
Image [224×224]
    → CLIP ViT-B/32 (frozen)
    → 50 patch vectors [50 × 768]  (hook on last ViT block)
    → patch projection layer
    → cross-attention sublayer in each of 12 GPT-2 blocks
    → text tokens query image patches at every generation step
```

### What Trains vs What Stays Frozen

| Component | Status |
|---|---|
| CLIP ViT-B/32 | Frozen |
| GPT-2 base weights | Frozen |
| MLP projection layer | **Trains** |
| LoRA A + B matrices (rank 16) | **Trains** |
| Cross-attention layers (Method B) | **Trains** |

---

## Setup

### Requirements

```bash
pip install "transformers==4.44.2"
pip install "peft==0.12.0"
pip install "accelerate==0.33.0"
pip install git+https://github.com/openai/CLIP.git
pip install pycocoevalcap pycocotools
pip install Pillow matplotlib tqdm
```

### Dataset

The project uses **COCO train2017** (v4) and **COCO val2017** (v1–v3).

- v4 downloads 20,000 images individually (~3.2GB) — no need to download the full 18GB zip
- All annotations are downloaded automatically from the official COCO CDN
- Images are cached to Google Drive; re-runs skip already-downloaded files

### Google Drive Folder Structure

```
MyDrive/
├── image-captioning/           ← v1
├── image-captioning-v2/        ← v2
├── image-captioning-v3/        ← v3
├── image-captioning-v4/        ← v4 (primary)
│   ├── data/train2017_subset/  ← 20k COCO images
│   ├── data/annotations/
│   ├── data/clip_embeddings_by_id.pt
│   ├── checkpoints/best_model_v4.pt
│   └── results/
└── image-captioning-method-b/  ← Method B
```

---

## Running the Notebooks

### v4 (Primary Model)

1. Open `image_captioning_v4.ipynb` in Google Colab
2. Runtime → Change runtime type → **H100 GPU** (recommended) or L4
3. Run all cells in order
4. Section 12 allows testing on your own images via URL or file upload

### Method B

1. Open `image_captioning_method_b.ipynb`
2. Requires H100 — patch extraction takes ~8 min, training ~7 min/epoch
3. Uses the same 20k COCO images as v4 (reused automatically)

### BLIP Comparison

1. Open `blip_comparison.ipynb`
2. Runs on any GPU — no training required
3. Evaluates pretrained BLIP on the same v4 test set

---

## Key Findings

**Architecture improvements dominate data scaling.**  
Going from a linear projection (v1) to MLP + Tanh with prefix=40 and LoRA rank 16 (v3) improved CIDEr from ~0.55 to 0.7508 on the same 5k dataset. Scaling to 100k pairs (v4) gave a further +0.016 — meaningful but smaller.

**v4 outperforms BLIP on COCO despite 1,290× less training data.**  
BLIP (trained on 129M pairs) achieves CIDEr 0.7311. Our v4 achieves 0.7668. Domain-specific LoRA fine-tuning on 100k COCO pairs outperforms a general-purpose pretrained model on the same benchmark.

**Cross-attention is data-hungry.**  
Method B provides richer spatial visual grounding (50 patch vectors vs 1 pooled vector) and produces qualitatively better captions on complex scenes, but the 12 randomly initialised cross-attention layers need substantially more data to converge. With 100k pairs, prefix conditioning outperforms cross-attention on all metrics.

**Contrastive decoding reduces hallucination.**  
Subtracting the language model's unconditional score (alpha=0.3–0.5) at each generation step penalises statistically likely but visually unsupported tokens without any retraining.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **BLEU-1/4** | N-gram overlap between generated and reference captions |
| **METEOR** | Rewards synonyms and paraphrase matches beyond exact overlap |
| **CIDEr** | TF-IDF weighted; rewards image-specific distinctive words — primary metric |
| **ROUGE-L** | Longest common subsequence between generated and reference |

---

## Decoding

All versions use **beam search k=5** with length penalty 1.2 for standard evaluation.  
**Contrastive decoding** (alpha=0.3–0.5) is available as an inference-time option to reduce hallucination.  
Method B uses manual beam search since GPT-2's built-in `generate()` does not support custom cross-attention layers.

---


## References

- Radford et al. (2021) — CLIP: Learning Transferable Visual Models from Natural Language Supervision
- Radford et al. (2019) — GPT-2: Language Models are Unsupervised Multitask Learners
- Hu et al. (2022) — LoRA: Low-Rank Adaptation of Large Language Models
- Lin et al. (2014) — Microsoft COCO: Common Objects in Context
- Li et al. (2022) — BLIP: Bootstrapping Language-Image Pre-training
- Li et al. (2023) — BLIP-2: Bootstrapping Language-Image Pre-training with Frozen LLMs
- Liu et al. (2024) — LLaVA: Visual Instruction Tuning
