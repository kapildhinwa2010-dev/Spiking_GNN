# Contributing

Thanks for your interest in NeuroSpiking!

## Setup
```bash
git clone https://github.com/kapildhinwa2010-dev/neurospiking.3D
cd neurospiking.3D
pip install torch numpy
pip install -e ".[dev]"   # or just: pip install pytest
pytest tests/ -v
```

## Workflow
1. Fork / branch from `main`
2. Keep changes focused; run `pytest tests/ -v` before pushing (CI runs it too)
3. Open a PR using the template; describe what changed and why

## Code style
- Small modules, explicit names, comments where the biology matters
- New modules need tests in `tests/`
- Keep CPU-friendly defaults so CI stays fast

## Good first issues
- More evaluations in `evaluate.py`
- Improving the 3D demo (`demo/index.html`)
- Real connectome experiments (`build_flywire_graph.py`)
