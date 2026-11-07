# Changelog

## [0.1.2] - 2026-11-07

### Added
- **NeuroSpiking-50M**: `model.py` trunk at hidden 1008 x 12 layers (~50.8M params), config in `configs/50m.json`
- **`train_50m.py`** — NEW training algorithm: **dopamine-gated 3-factor local learning** (`dw = lr * reward * eligibility`, no backprop). Uses ALL modules: model, memory (MembraneCache across samples), tokenizer (topic corpus -> co-occurrence graph), homeostasis (threshold adaptation), stdp (plasticity index), dopamine (reward semantics)
- **Validation discovery**: zero-reward errors pe network 0% accuracy pe atki rehti hai — low-dopamine error signal (+0.25) hi learning enable karta hai
- **CI training workflow** (`train_50m.yml`): 15-min budget training on free GitHub Actions runner, results JSON + README table auto-commit (`[skip ci]`)
- `scripts/update_readme_results.py`, `TRAINING_50M.md` (model/data/training docs), `tests/test_train_50m.py` (4 tests), `make train-50m`

## [0.1.1] - 2026-11-07

### Added
- **`dopamine.py`** — Reward-Modulated (3-factor) learning: `dw = lr * reward * eligibility`. Biological credit assignment: reward only strengthens recently-paired synapses (all 5 behaviors validated before shipping)
- **Automation (no manual steps left)**:
  - Release workflow: tags + publishes a GitHub Release automatically when `CHANGELOG.md` changes
  - Pages workflow: deploys the 3D demo to GitHub Pages automatically when `demo/` changes
  - `scripts/download_weights.py` + `make weights`: one-command download of all pre-trained models from Drive
- **`Makefile`** — one-command setup: `make install / test / lint / eval / train / weights / demo`
- Tests: `tests/test_dopamine.py` (6 tests)

### Changed
- Version bump to 0.1.1; README updated (dopamine module, automation, scripts/Makefile in structure)

## [0.1.0] - 2026-11-07

First production-ready release of NeuroSpiking.

### Added
- **Architectures**: `model.py` (241M), `model_500m.py` (~503M), `configs/241m.json|500m.json|800m.json` (800M sized by scaling estimate)
- **Spiking memory**: `MembraneCache` (KV-cache analog - membranes persist across forward calls), `SpikingMemory`, `MultiTimescaleMemory` (fast + slow traces), `EligibilityTrace`
- **Tokenizer**: `tokenizer.py` (vocab.json, <unk>, text -> graph)
- **New training way**: `train_memory.py` - e-prop style local online learning vs BPTT (validated: both reach ~0 loss on a memory task)
- **Biological modules**: `stdp.py` (spike-timing dependent plasticity), `homeostasis.py` (adaptive thresholds, anti-seizure)
- **Evaluation**: `evaluate.py` (sparsity, hub suppression, lesion study, semantic proxy)
- **Data**: `build_flywire_graph.py` (experimental real connectome pipeline)
- **Interactive 3D demo**: `demo/index.html` (three.js; validated LIF core with E/I balance and refractory periods)
- **Tests + CI**: 12 pytest files, GitHub Actions (pytest + advisory pyflakes)
- **Packaging**: `pyproject.toml` v0.1.0, CONTRIBUTING.md, MODEL_CARD.md, issue/PR templates

### Changed
- File renames for runnability (`modle.py` -> `model.py`, `Inference.py` -> `inference.py`, `requirement.txt` -> `requirements.txt`, `licence` -> `LICENSE`)
- 500M single cell (594 lines) modularized into `graph_builder.py`, `model_500m.py`, `train_utils.py`
- `train.py` saves `vocab.json` + `config.json`; `inference.py` shows vocab words

### Fixed
- `!pip install` notebook line that crashed `python train_500m.py`
- README structure and commands now match the actual files
