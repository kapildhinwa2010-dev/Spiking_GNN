# ============================================================
# TRAINING UTILITIES (500M pipeline)
# Extracted from the original 500M training cell:
#   - make_optimizer  : bitsandbytes AdamW8bit with AdamW fallback (section 7)
#   - get_save_dir    : Google Drive mount helper, Colab only (section 2)
#   - save_checkpoint : checkpoint writer, was save_model (section 9)
# ============================================================

import os

import torch


def make_optimizer(model, lr=1e-4):
    """8-bit AdamW (bitsandbytes) with normal AdamW fallback (extracted from 500M cell)."""

    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )
        print("✅ Using bitsandbytes AdamW8bit optimizer")
        return opt, "AdamW8bit"
    except Exception as e:
        print(f"⚠️ bitsandbytes 8-bit optimizer failed: {e}")
        print("⚠️ Falling back to normal AdamW. VRAM pressure badh sakti hai.")
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,
            foreach=False,
        )
        return opt, "AdamW"


def get_save_dir(use_google_drive=True, local_dir="brainai_500m"):
    """Google Drive mount (Colab only) with local fallback (extracted from 500M cell)."""
    if use_google_drive:
        try:
            from google.colab import drive
            if not os.path.exists("/content/drive/MyDrive"):
                drive.mount("/content/drive")
            save_dir = "/content/drive/MyDrive/BrainAI_500M"
        except Exception as e:
            print(f"⚠️ Drive mount failed: {e}")
            save_dir = local_dir
    else:
        save_dir = local_dir

    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 Save directory: {save_dir}")
    return save_dir


def save_checkpoint(model, save_dir, cfg, train_cfg, params, trained_steps, filename="brain_spiking_500m.pth"):
    """Save a full training checkpoint (extracted from save_model in the 500M cell)."""
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "train_config": train_cfg,
        "params": params,
        "trained_steps": trained_steps,
        "model_name": "SpikingGraph3D_500M_KapilDhinwa",
    }

    path = os.path.join(save_dir, filename)

    print(f"\n💾 Saving 500M checkpoint to: {path}")
    print("⏳ This may take some time because file is ~2 GB.")

    torch.save(ckpt, path)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"✅ Saved! File size: {size_mb:.1f} MB")

    return path
