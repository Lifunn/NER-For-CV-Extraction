import spacy
import wandb
import sys

def train_with_wandb_config():
    """
    Fungsi untuk menjalankan spaCy train dengan konfigurasi dari W&B sweep.
    """
    # Inisialisasi run W&B.
    # Agent W&B akan secara otomatis mengisi wandb.config dengan hiperparameter
    # untuk trial saat ini.
    wandb.init()
    
    config_path = "config.cfg" # Path ke config dasar Anda

    # Ambil hiperparameter dari W&B dan siapkan sebagai override untuk spaCy
    overrides = {
        "training.dropout": wandb.config["training.dropout"],
        "training.optimizer.learn_rate": wandb.config["training.optimizer.learn_rate"],
        "training.batcher.size": wandb.config["training.batcher.size"],
        "components.ner.model.hidden_width": wandb.config["components.ner.model.hidden_width"]
    }
    
    # Jalankan spaCy train secara programatik
    try:
        spacy.cli.train(
            config_path,
            overrides=overrides,
            # Jika Anda menggunakan --output di command line, spaCy akan membuat subfolder
            # untuk setiap run di sana, yang sangat membantu.
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        # Memberitahu W&B bahwa run ini gagal
        wandb.finish(exit_code=1) 
        sys.exit(1) # Keluar dari skrip jika training gagal

if __name__ == "__main__":
    train_with_wandb_config()