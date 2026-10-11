# 🧠 NeuroSpiking-241M: God-Tier Neuromorphic Graph AI

A 241 Million parameter Spiking Graph Neural Network trained on real **Flywire Connectome data** (incorporating Neurotransmitters like Dopamine, GABA, Serotonin) and **10 Million Language Tokens**. 

This project pushes the boundaries of Neuromorphic Computing by demonstrating biological plausibility, including emergent **Sensory Gating** (noise suppression) and **Lesion-induced Disinhibition** (AI seizure states).

## ✨ Key Features
- **Massive Scale:** 241,204,234 parameters with 8 Spiking Layers and 16 Attention Heads.
- **Neuro-Transmitter Aware:** Edges are weighted by real biological chemicals (GABA, Glutamate, etc.).
- **Biological Plausibility:** Exhibits sparse firing, E/I (Excitation/Inhibition) balance, and hub suppression.
- **Memory Efficient:** Trained on a single 16GB GPU (NVIDIA T4) using advanced Gradient Checkpointing.
- **Dual-Modality:** Trained on both 3D Spatial Brain Graphs and 10M Text Co-occurrence Graphs.

## 📊 Official Benchmarks
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Parameters** | 241.2 M | God-Tier Scale |
| **Throughput** | ~835 TPS | Tokens Per Second (Inference) |
| **Latency** | ~1197 ms | Per 1000-node graph |
| **Semantic Dist** | 0.7089 | 3D Contextual Clustering |
| **Hub Suppression**| 1.0410 | Sensory Gating (Ignoring 'the', 'and') |
| **Lesion Impact** | -20.6% | Disinhibition (Seizure state on hub damage) |

## 🏗️ Architecture
- **Neuron Model:** Sigmoid-based Surrogate LIF (Leaky Integrate-and-Fire) with learnable decay and threshold.
- **Spatial Conv:** Multi-Head Edge-Aware Graph Convolution with 3D spatial + 6 chemical features.
- **Optimization:** AdamW with Cosine Annealing LR and Gradient Clipping.

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Downloads the `text8` corpus (10M tokens), builds the language graph, and trains the 241M model.
```bash
python train.py
```

### 3. Run Benchmarks & Lesion Study
Loads the trained model and runs the speed test, semantic clustering, and brain lesion simulation.
```bash
python benchmark.py
```

## 📂 Project Structure
- `model.py`: Core PyTorch architecture (Spiking Neurons & Graph Convs).
- `train.py`: Data pipeline, graph construction, and training loop.
- `benchmark.py`: Inference speed, semantic tests, and lesion study.

## 📜 License
MIT License. Free for research and educational purposes.