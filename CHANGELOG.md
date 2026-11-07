# Changelog

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
