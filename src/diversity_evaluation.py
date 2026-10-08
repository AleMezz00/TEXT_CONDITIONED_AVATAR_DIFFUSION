import json
import torch
import torch.nn as nn

from pathlib import Path
from datasets import load_dataset
from torchvision.transforms.functional import pil_to_tensor

import avatar_generation as generation

from preprocessing import DATASET_NAME, DATASET_CONFIG, generate_caption

# DEFINISCO I PERCORSI PRINCIPALI
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_PATH = DATA_DIR / "splits.json"

# CARTELLA PER I RISULTATI DELLE MODEL METRICS
RESULTS_DIR = PROJECT_ROOT / "model_metrics_results"
DIVERSITY_RESULTS_DIR = RESULTS_DIR / "diversity"
DIVERSITY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CARICO GLI SPLIT
with open(SPLITS_PATH, "r") as file:
    splits = json.load(file)

# CARICO IL DATASET ORIGINALE
original_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")

# CARICO IL MODELLO CONDITIONAL ADDESTRATO
vocabulary, tokenizer_config, text_encoder, unet, ddpm = (generation.load_generation_components())

# PRENDO SOLO IL PRIMO PROMPT DAL TEST IID
sample = original_dataset[splits["test_iid"][0]]
prompt = generate_caption(sample)

# DEFINISCO UN VETTORE CON CINQUE SEED DIFFERENTI
DIVERSITY_SEEDS = [42, 43, 44, 45, 46]

generated_tensors_list = []

for seed in DIVERSITY_SEEDS:

    generated_image = generation.generate_avatar(
        prompt=prompt,
        seed=seed,
        vocabulary=vocabulary,
        tokenizer_config=tokenizer_config,
        text_encoder=text_encoder,
        unet=unet,
        ddpm=ddpm
    )

    # SALVO L'IMMAGINE GENERATA
    output_path = DIVERSITY_RESULTS_DIR / f"seed_{seed}.png"
    generated_image.save(output_path)

    # CONVERTO L'IMMAGINE IN TENSORE PER IL CALCOLO DELLA DIVERSITA'
    generated_tensor = pil_to_tensor(generated_image).float() / 255.0
    generated_tensors_list.append(generated_tensor)

# DEFINISCO LA MSE LOSS E IL VETTORE DEI CONFRONTI
mse_loss = nn.MSELoss()
pairwise_mse_values = []

# CALCOLO LA MSE TRA TUTTE LE COPPIE DI IMMAGINI
for first_index in range(len(generated_tensors_list)):
    for second_index in range(first_index + 1, len(generated_tensors_list)):

        pairwise_mse = mse_loss(generated_tensors_list[first_index], generated_tensors_list[second_index])
        pairwise_mse_values.append(pairwise_mse.item())

# CALCOLO LA DIVERSITA' MEDIA
average_diversity_mse = sum(pairwise_mse_values) / len(pairwise_mse_values)

print(f"\nPrompt: {prompt}")
print(f"Seeds: {DIVERSITY_SEEDS}")
print(f"Average pairwise diversity MSE: {average_diversity_mse:.6f}")