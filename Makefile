# NeuroSpiking one-command setup
.PHONY: install test lint eval train train-50m weights demo

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt
	pip install pytest pyflakes gdown

test:
	pytest tests/ -v

lint:
	pyflakes model.py model_500m.py memory.py tokenizer.py stdp.py dopamine.py homeostasis.py graph_builder.py train_utils.py evaluate.py train_memory.py train_50m.py || true

eval:
	python evaluate.py --hidden 256 --nodes 120

train:
	python train.py

train-50m:
	python train_50m.py --budget-min 15

weights:
	python scripts/download_weights.py

demo:
	python -c "import webbrowser, os; webbrowser.open('file://' + os.path.abspath('demo/index.html'))"
