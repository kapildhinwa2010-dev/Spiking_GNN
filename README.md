# 🧠 NeuroSpiking-241M: God-Tier Neuromorphic Graph AI

**Author:** Kapil Dhinwa  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16qU6PoYAcMOywR7e0t3mRs7mRpcXyjkP)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/kapildhinwa2010-dev/neurospiking.3D/actions/workflows/tests.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/neurospiking.3D/actions/workflows/tests.yml)

A **241 Million parameter** Spiking Graph Neural Network trained on real **Flywire Connectome data** (incorporating Neurotransmitters like Dopamine, GABA, Serotonin) and **10 Million Language Tokens**. 

This project pushes the boundaries of Neuromorphic Computing by demonstrating biological plausibility, including emergent **Sensory Gating** (noise suppression) and **Lesion-induced Disinhibition** (AI seizure states).

---

## 🔗 Model Zoo & Downloads

We have trained the exact same 241M architecture on two completely different scientific domains. Download the pre-trained weights from Google Drive:

| Model Name | Domain / Training Data | File Name | Download Link |
| :--- | :--- | :--- | :--- |
| 🧠 **Bio-Brain** | Flywire Connectome (Real Synapses + Neurotransmitters) | `flywire_bio_brain_241m.pth` | [Download Bio-Model]( https://drive.google.com/file/d/13lXRsiAr6qiLvJmhrEjV5hFLf-_EC9i2/view?usp=drivesdk) |
| 📚 **Language-Brain** | 10M Text Tokens (Wikipedia Co-occurrence Graph) | `language_cognitive_brain_241m.pth` | [Download Lang-Model](https://drive.google.com/file/d/1yDnHIzO0qIJfbBS4-N0Gahfzaga7WnLu/view?usp=drivesdk) |
| ⚙️ **Configs & Vocab** | Architecture Configs & 3000 Word Vocabulary | `config.json`, `vocab.json` | [Download Configs](👉 https://drive.google.com/file/d/1YHcUmBE7cpChqA2DaMIDFisyTTib0hrV/view?usp=drivesdk) |
 scroll for links


---

## ✨ Key Features
- **Massive Scale:** 241,204,234 parameters with 8 Spiking Layers and 16 Attention Heads.
- **Neuro-Transmitter Aware:** Edges are weighted by real biological chemicals (GABA, Glutamate, Dopamine, etc.).
- **Biological Plausibility:** Exhibits sparse firing, E/I (Excitation/Inhibition) balance, and hub suppression.
- **Memory Efficient:** Trained on a single 16GB GPU (NVIDIA T4) using advanced **Gradient Checkpointing** (saves 80% VRAM).
- **Dual-Modality:** Proven effective on both 3D Spatial Brain Graphs and 10M Text Co-occurrence Graphs.

---

## 📊 Official Benchmarks

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Parameters** | 241.2 M | God-Tier Scale |
| **Throughput** | ~835 TPS | Tokens Per Second (Inference) |
| **Latency** | ~1197 ms | Per 1000-node graph |
| **Semantic Dist** | 0.7089 | 3D Contextual Clustering (e.g., King-Queen) |
| **Hub Suppression**| 1.0410 | Sensory Gating (Ignoring noise like 'the', 'and') |
| **Lesion Impact** | -20.6% | Disinhibition (Seizure state on hub damage) |

---

## 🔬 Evaluation & Reproducibility

```bash
python evaluate.py                        # full-size (config.json dims)
python evaluate.py --hidden 256 --nodes 120  # quick CPU run
```

```evaluate.py``` computes sparsity, hub suppression, lesion/disinhibition and (if ```vocab.json``` exists) semantic clustering, then writes ```evaluate_results.json```.

Real connectome data (experimental, needs a Flywire token from https://globalv1.flywire.ai/auth):

```bash
export FLYWIRE_TOKEN="your_token"
python build_flywire_graph.py   # saves flywire_graph.pt with real neurotransmitter features
```

## 🧪 Tests & CI

```bash
pip install pytest
pytest tests/ -v
```

GitHub Actions runs the test suite (CPU PyTorch) on every push and pull request to ```main```.

---

## 🧠 Memory, KV-Cache & Tokenizer

Spiking networks have a natural memory: the **membrane potential**. This repo now ships:

- **`tokenizer.py`** — word-level tokenizer: loads/saves `vocab.json`, `<unk>` handling, text → co-occurrence graph edges
- **`memory.py`** — the spiking **KV-cache**: `MembraneCache` + `forward_cached()` keep membrane potentials **across** forward calls (the brain keeps thinking instead of resetting every call); `SpikingMemory` (working-memory trace + top-k recall); `EligibilityTrace` (e-prop style local learning)
- **`train_memory.py`** — a **new training way**: **local eligibility-trace (e-prop style) online learning — no backprop through time** — vs standard surrogate-gradient BPTT, on a delayed match-to-sample task that requires memory across 6 silent steps. In validation runs both ways reached ~0 loss (100% pattern match).

```python
from tokenizer import WordTokenizer
from memory import MembraneCache, forward_cached

tok = WordTokenizer.load("vocab.json")
cache = MembraneCache(num_nodes=len(tok), hidden_dim=3072, num_layers=8)
rates = forward_cached(model, x, edge_index, edge_attr, cache)  # membranes persist between calls
```

```bash
python train_memory.py   # runs the e-prop vs BPTT comparison, writes train_memory_results.json
```

---

## 🏗️ Architecture
- **Neuron Model:** Sigmoid-based Surrogate LIF (Leaky Integrate-and-Fire) with learnable decay and threshold.
- **Spatial Conv:** Multi-Head Edge-Aware Graph Convolution with 3D spatial + 6 chemical features.
- **Optimization:** AdamW with Cosine Annealing LR and Gradient Clipping.

---

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Pre-trained Weights
Download your preferred model (`.pth` file) from the **Model Zoo** links above and place it in the root directory. `config.json` is already included in the repo, and `train.py` automatically saves `vocab.json` and `config.json` for you.

### 3. Run Inference (Stimulate the Brain)
```bash
python inference.py
```

### 4. Train from Scratch (Optional)
Downloads the `text8` corpus (10M tokens), builds the language graph, and trains the 241M model.
```bash
python train.py
```

---

## 📂 Project Structure
```text
├── model.py            # Core PyTorch architecture (Spiking Neurons & Graph Convs)
├── train.py            # Data pipeline, graph construction, and 241M training loop
├── train_500m.py       # 500M parameter variant (8-bit optimizer, Drive checkpointing)
├── train_memory.py     # NEW training way: e-prop local online learning vs BPTT
├── evaluate.py         # Sparsity, hub suppression, lesion & semantic metrics
├── build_flywire_graph.py # (Experimental) Real Flywire connectome -> flywire_graph.pt
├── tokenizer.py        # Word tokenizer: vocab.json, encode/decode, text -> graph
├── memory.py           # Spiking memory: MembraneCache (KV-cache analog), working memory
├── benchmark.py        # Inference speed benchmark
├── inference.py        # Interactive CLI to stimulate brain nodes
├── dormant_recruitment.py # Research suite: dormant neuron recruitment + ablations
├── teaching_cell.py    # Full teaching cell (Hindi comments, 20 core concepts)
├── config.json         # Architecture config used by inference.py / evaluate.py
├── requirements.txt    # Python dependencies
├── tests/              # Pytest suite (run by GitHub Actions on every push)
├── .github/workflows/  # CI: tests on push/PR to main
├── LICENSE             # MIT
├── .gitignore          # Ignores heavy .pth files and datasets
└── README.md           # Project documentation
```

---

## 🤝 Contributions & Acknowledgments

**Project Engineered & Directed by:** Kapil Dhinwa  
**Architecture & Codebase:** Coded by AI  

*This research project was conceptualized by Kapil Dhinwa, with the underlying PyTorch architecture, memory optimization (Gradient Checkpointing), and debugging heavily assisted and coded by AI. The goal was to push the limits of Neuromorphic AI on constrained hardware (Colab T4 16GB).*

---

## 📜 License
MIT License. Free for research, educational, and commercial purposes.