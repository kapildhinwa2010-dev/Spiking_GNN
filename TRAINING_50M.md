# NeuroSpiking-50M — Model, Data & Training (CI-Trained)

> Yeh document batata hai ki 50M model kaise banaya, training data kya tha,
> aur training kaise hui. Training GitHub Actions par chalti hai (15-min budget)
> aur results khud README me aa jaate hain — koi manual step nahi.

## 1. Model kaise banaya

Same architecture jo 241M/500M me hai (`model.py` ka `WholeBrainSpikingNet`), scaled down:

| | 241M | **50M (yeh)** |
| :--- | :--- | :--- |
| hidden_dim | 3072 | **1008** |
| num_layers | 8 | **12** |
| num_heads | 16 | 16 |
| timesteps | 4 | 2 |
| params | 241.2M | **~50.8M** |

Har layer me: multi-head edge-aware graph conv (3D + 10-dim chemical edge features) + surrogate-gradient LIF neuron (learnable decay + threshold).

## 2. Training data

- **Corpus:** 10 neuroscience topics (brain, memory, learning, sleep, vision, language, emotion, movement, senses, AI) — template sentences, fully deterministic (seeded).
- **Tokenizer:** `tokenizer.py` → ~110-word vocab (vocab.json format se consistent).
- **Graph:** poore corpus ka co-occurrence graph (window=5), top-80 undirected pairs → 160 directed edges; 10-dim deterministic edge features (sin/cos positional + co-occurrence count).

Task: sentence → topic classification (10 classes). Node features = us sentence ke word counts.

## 3. Training algorithm (NAYA): Dopamine-Gated 3-Factor Local Learning

50M trunk **bina backprop** chalta hai (features = mean spike rates), aur sirf local rules se learn hota hai:

`python
dw = lr * reward * eligibility
eligibility = outer(onehot - softmax, features)   # readout delta
reward = +1.0   # prediction sahi -> dopamine high
reward = +0.25  # prediction galat -> low dopamine (par zero NAHI)
`

**Validation discovery:** simulation me prove hua ki agar galat predictions par dopamine **zero** ho to network kuch nahi seekhta (0% accuracy) — low-dopamine error signal hi learning enable karta hai. Yeh biological literature se match karta hai (dopamine dips bhi informative hote hain).

Saath me:
- **MembraneCache** (`memory.py`): membrane state samples ke across persist karti hai — spiking "KV-cache", network context yaad rakhta hai
- **HomeostaticRegulator** (`homeostasis.py`): har 8 samples neuron thresholds ko target rate (0.3) ki taraf adapt karta hai — anti-seizure
- **STDPLearner** (`stdp.py`): co-activation plasticity index (analysis metric)

## 4. CI Training (GitHub Actions)

Workflow `.github/workflows/train_50m.yml`:

1. 15-minute budget ke saath `python train_50m.py --budget-min 15` chalata hai (free 2-core CPU runner)
2. `scripts/update_readme_results.py` results table README me daalta hai
3. `train_50m_results.json` + updated README auto-commit ho jaate hain (`[skip ci]`)

## 5. Khud run karo

`bash
python train_50m.py --budget-min 2      # 2-min quick local run
python train_50m.py --budget-min 15     # CI jaisa full run
# ya Actions tab me "Train 50M model" workflow manually dispatch karo
`

Config: `configs/50m.json`. Results: `train_50m_results.json` + README section.
