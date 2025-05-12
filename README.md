

# 🧪 MolT5 Playground: Graph-based Multimodal Chemistry LLM

**MolT5 Playground** is an experimental framework for training and evaluating multimodal models that bridge **chemical structure (graphs)** and **natural language**.  
It combines Graph Neural Networks (GNNs) with Large Language Models (LLMs) for molecule-level tasks like captioning, embedding, and generation.

> ⚠️ This is a **prototype research project**. The performance is limited and the work is incomplete — see the roadmap below.

---

## 🧬 What Is This Project?

This repo explores lightweight implementations of **chemistry-focused multimodal models** based on:

- Graph encoders (GIN / DGL)  
- Language models (MiniLM / TinyLlama)  
- CLIP-style contrastive learning  
- Instruction-tuned caption generation  

It's inspired by recent Graph-LLMs like **Mol-LLaMA**, **LlaMo**, and **ChemLLM**, but aims to keep things minimal, inspectable, and Colab-friendly.

---

## ✨ Highlights

| Feature | Description |
|--------|-------------|
| 🧠 **Graph × Language** | SMILES + captions → GNN + LLM fusion |
| 🔀 **SMILES Augmentation** | Randomized SMILES for robust training |
| 📊 **CLIP-style Contrastive Learning** | InfoNCE loss with momentum queue |
| 💬 **ChatML Prompt Injection** | Embed molecule tokens into chat-style LLMs |
| 💻 **Colab-Ready** | Lightweight enough for free/Pro Colab tiers |

---

## 🧪 Models Included

### 1. `CLIPMol`: GNN + BERT

- Vision: `gin_supervised_contextpred` + MHA + projection
- Text: `all-MiniLM-L6-v2` (Sentence-BERT)
- Objective: contrastive embedding alignment (InfoNCE)

### 2. `MiniMolTiny`: GNN + TinyLlama Chat

- Language model: `TinyLlama-1.1B-Chat-v0.4` (frozen)
- Inserts molecule embeddings into `<mol_patch_i>` token positions
- Generates natural language captions from graph input

---

## 🧪 Sample Experiment (Stage 1)

| Epoch | Diagonal Sim | Recall@1 | Recall@5 |
|-------|---------------|----------|----------|
| 1     | 0.453         | 0.071    | 0.232    |
| 6     | 0.713         | 0.311    | 0.628    |
| 12    | 0.742         | 0.357    | 0.699    |

> ⚙️ Trained with `n_aug=3` randomized SMILES, batch size 32, gradient accumulation 8×

**Observations**:
- Stable training behavior
- CLIP contrastive loss effectively learns graph-text alignment
- Text encoder (MiniLM) is a bottleneck — upgrade needed

---

## 🛠️ Setup

### Colab / Local

```bash
# Python 3.9+
pip install -r requirements.txt
```

Supports:
- Google Colab (GPU/TPU)
- VSCode DevContainer (Docker)
- Local GPU setups (Linux/Windows)

---

## 🗂️ Directory Structure

```
molt5-project/
├── molt5_core/            # Core model classes (GNNs, LLMs)
├── scripts/               # Training entrypoints
├── configs/               # YAML config files for experiments
├── utils/                 # Logging, helpers
├── notebooks/             # Colab / Jupyter notebooks (optional)
├── requirements.txt
└── README.md
```

---

## 🔭 Limitations

This is **not a production model**. Limitations include:

- ⚠️ **Limited accuracy**: current BLEU and recall scores are low  
- 📉 **Small training dataset**: ~26k molecules (ChEBI-20)  
- 🧩 **Data bottleneck**: real generalization requires better and broader graph-text datasets  
- ❌ **No structure comparison yet**: current model handles isolated molecules only

---

## 🚧 Research Notes & Future Plans

> _"It feels like a niche. Not a big-issue area."_  
> _"The goal should be **understanding chemical transformations**, not just captioning."_  
> _"LLMs are strong at language, but not graph-local reasoning. That's our advantage."_

✅ **Next Steps**:
- Pretrain encoders on larger unlabeled molecule sets (e.g., PubChem SMILES + IUPAC)  
- Caption finetuning on curated data (ChEBI, patents, etc.)  
- Move toward **pairwise tasks** (e.g., reaction understanding, graph comparisons)  
- Explore **better base models** (Mistral, Phi-2) for language generation  

---

## 📚 Related Work

- [MolT5 (Blender NLP)](https://github.com/blender-nlp/MolT5)
- [Mol-LLaMA](https://arxiv.org/abs/2403.07954)
- [LlaMo: LLaMA for Molecules](https://arxiv.org/abs/2402.16655)
- [ChemLLM](https://arxiv.org/abs/2306.05445)

---

## 📄 License

This project is licensed under the MIT License.  
See `LICENSE` for details.

---

## 🙋 Contributions Welcome

This is a **work-in-progress** project.  
If you’re exploring similar problems or want to build on this idea, feel free to open issues or PRs. Feedback is always appreciated!
```
