"""Download all pre-trained NeuroSpiking models from Google Drive.

Usage:
    pip install gdown
    python scripts/download_weights.py          # all files
    python scripts/download_weights.py --only bio
"""
import argparse
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FILES = {
    "bio": ("13lXRsiAr6qiLvJmhrEjV5hFLf-_EC9i2", "flywire_bio_brain_241m.pth"),
    "lang": ("1yDnHIzO0qIJfbBS4-N0Gahfzaga7WnLu", "language_cognitive_brain_241m.pth"),
    "configs": ("1YHcUmBE7cpChqA2DaMIDFisyTTib0hrV", "configs_vocab.zip"),
}


def main():
    try:
        import gdown
    except ImportError:
        sys.exit("gdown not installed. Run: pip install gdown")

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(FILES), default=None)
    args = parser.parse_args()

    targets = [args.only] if args.only else list(FILES)
    for key in targets:
        file_id, out = FILES[key]
        print(f"Downloading {key} -> {out} ...")
        gdown.download(id=file_id, output=out, quiet=False)
    print("Done. Weights are in the repo root, ready for inference.py / evaluate.py.")


if __name__ == "__main__":
    main()
