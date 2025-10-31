"""
ner_training.py
---------------
Train a SpaCy Transformer-based NER model for CV entity extraction.

This script supports both:
1. Direct training (SpaCy CLI)
2. Weights & Biases (W&B) hyperparameter sweep for optimization
"""

import os
import subprocess
import wandb
from spacy.cli.train import train as spacy_train

# === Configurations ===
PROJECT_NAME = "NER-CV-Extraction"
CONFIG_PATH = "./configs/spacy_config.cfg"
OUTPUT_DIR = "./output"
TRAIN_PATH = "./corpus/train_data.spacy"
DEV_PATH = "./corpus/test_data.spacy"
USE_GPU = True  # Automatically use GPU if available
RUN_SWEEP = True  # Set to False to skip W&B sweep and just train directly

# === Ensure directories exist ===
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_direct():
    """
    Run standard SpaCy training using CLI equivalent.
    """
    print("🚀 Starting direct SpaCy training...")
    cmd = [
        "python", "-m", "spacy", "train",
        CONFIG_PATH,
        "--output", OUTPUT_DIR,
        "--paths.train", TRAIN_PATH,
        "--paths.dev", DEV_PATH,
    ]
    if USE_GPU:
        cmd += ["--gpu-id", "0"]

    subprocess.run(cmd, check=True)
    print("\n✅ Training complete. Model saved in:", OUTPUT_DIR)


def train_sweep():
    """
    Execute one W&B sweep training iteration.
    Mirrors the logic from the original Colab notebook.
    """
    print("🚀 Starting W&B sweep run...")
    with wandb.init(project=PROJECT_NAME) as run:
        overrides = {**wandb.config}
        overrides["training.logger.project_name"] = run.project

        print("--- Active Sweep Parameters ---")
        for k, v in overrides.items():
            print(f"{k}: {v}")

        try:
            spacy_train(
                CONFIG_PATH,
                OUTPUT_DIR,
                overrides=overrides,
                use_gpu=0 if USE_GPU else -1
            )
        except Exception as e:
            print(f"❌ Training failed: {e}")
        finally:
            print("--- Run finished ---")


def launch_sweep():
    """
    Launch a W&B Bayesian sweep using `configs/config.yaml`.
    """
    print("🧠 Initializing W&B sweep...")
    sweep_id = wandb.sweep("configs/config.yaml", project=PROJECT_NAME)
    print(f"Sweep created with ID: {sweep_id}")
    wandb.agent(sweep_id, function=train_sweep, count=10)
    print("\n✅ Sweep completed.")


def main():
    if RUN_SWEEP:
        launch_sweep()
    else:
        train_direct()


if __name__ == "__main__":
    main()
