# Model Card - NeuroSpiking

## Model
Spiking graph neural networks over 3D brain-like graphs (surrogate-gradient LIF neurons, edge-aware multi-head graph convolution with spatial + chemical edge features).
- **241M**: hidden 3072, 8 layers, 16 heads, 4 timesteps
- **500M**: hidden 3648, 12 layers, 16 heads, 3 timesteps
- **800M**: hidden 3968, 16 layers, 16 heads, 3 timesteps (config provided; sized by scaling estimate)

## Training Data
- **Language-Brain**: text8 corpus (10M tokens), top-3000 word co-occurrence graph
- **Bio-Brain**: Flywire connectome (experimental; `build_flywire_graph.py` downloads real synapses + neurotransmitter predictions)

## Intended Use
Research and education in neuromorphic computing, spiking networks and biologically plausible learning (STDP, e-prop, homeostasis). Not a production language model.

## Limitations (honest)
- Training runs so far are short (tens of steps on a free Colab T4); benchmark numbers in the README come from those Colab runs - reproduce with `evaluate.py`
- Until the Flywire pipeline is run, 6 of the 10 edge features in the synthetic pipeline are placeholders, not real neurotransmitter data
- e-prop local learning is demonstrated on a small network (`train_memory.py`); scaling it to 241M+ is future work

## License
MIT - Kapil Dhinwa
