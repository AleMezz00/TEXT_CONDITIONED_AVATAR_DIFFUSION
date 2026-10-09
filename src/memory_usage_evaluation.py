import json
import torch

from pathlib import Path
from datasets import load_dataset

import avatar_generation as generation

from preprocessing import DATASET_NAME, DATASET_CONFIG, generate_caption

# DEFINISCO I PERCORSI PRINCIPALI
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_PATH = PROJECT_ROOT / "data" / "splits.json"

# CARICO GLI SPLIT
with open(SPLITS_PATH, "r") as file:
    splits = json.load(file)

# CARICO IL DATASET ORIGINALE
original_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")

# CARICO IL MODELLO CONDITIONAL ADDESTRATO
vocabulary, tokenizer_config, text_encoder, unet, ddpm = (generation.load_generation_components())

# UTILIZZO UN PROMPT DEL TEST IID
sample = original_dataset[splits["test_iid"][0]]
prompt = generate_caption(sample)

# AZZERO LE STATISTICHE DEL PICCO DI MEMORIA GPU
torch.cuda.reset_peak_memory_stats()

# GENERO UN AVATAR
generation.generate_avatar(
    prompt=prompt,
    seed=42,
    vocabulary=vocabulary,
    tokenizer_config=tokenizer_config,
    text_encoder=text_encoder,
    unet=unet,
    ddpm=ddpm
)

# SINCRONIZZO LA GPU PRIMA DI LEGGERE IL RISULTATO
torch.cuda.synchronize()

# RECUPERO IL PICCO MASSIMO DI MEMORIA GPU ALLOCATA
peak_memory_bytes = torch.cuda.max_memory_allocated()

# CONVERTO IL VALORE DA BYTE A MEGABYTE
peak_memory_mb = peak_memory_bytes / (1024 ** 2)

print(f"Peak GPU memory usage: {peak_memory_mb:.2f} MB")