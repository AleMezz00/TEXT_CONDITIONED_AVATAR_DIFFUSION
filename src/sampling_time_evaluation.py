import json
import time
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

# UTILIZZO SOLO IL PRIMO SAMPLE DEL TEST IID
sample = original_dataset[splits["test_iid"][0]]
prompt = generate_caption(sample)

# SPECIFICO 3 SEED PER LA GENERAZIONE DI 3 AVATAR DIFFERENTI
SAMPLING_SEEDS = [42, 43, 44]

sampling_times_list = []

for seed in SAMPLING_SEEDS:

    # SINCRONIZZO LA GPU PRIMA DI INIZIARE LA MISURAZIONE
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    generation.generate_avatar(
        prompt=prompt,
        seed=seed,
        vocabulary=vocabulary,
        tokenizer_config=tokenizer_config,
        text_encoder=text_encoder,
        unet=unet,
        ddpm=ddpm
    )

    # SINCRONIZZO LA GPU PRIMA DI TERMINARE LA MISURAZIONE
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    sampling_time = end_time - start_time
    sampling_times_list.append(sampling_time)

    print(f"Seed {seed} sampling time: {sampling_time:.4f} seconds")

# CALCOLO IL TEMPO MEDIO DI SAMPLING
average_sampling_time = sum(sampling_times_list) / len(sampling_times_list)

print(f"\nAverage sampling time: {average_sampling_time:.4f} seconds")