import json
import torch
import torch.nn as nn

from pathlib import Path
from datasets import load_dataset

import avatar_generation as generation

from preprocessing import DATASET_NAME, DATASET_CONFIG
from dataset import CartoonAvatarDataset, create_dataloader
from unet import sinusoidal_time_embedding

# DEFINISCO I PERCORSI PRINCIPALI
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_PATH = DATA_DIR / "splits.json"

# DEFINISCO IL PERCORSO E CREO LA CARTELLA PER I RISULTATI DELLA VALUTAZIONE QUANTITATIVA
QUANTITATIVE_RESULTS_DIR = PROJECT_ROOT / "quantitative_results"
QUANTITATIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CARICO GLI SPLIT SALVATI
with open(SPLITS_PATH, "r") as file:
    splits = json.load(file)

# CARICO IL DATASET ORIGINALE
original_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG,split="train")

# CARICO IL MODELLO CONDITIONAL GIA' ADDESTRATO
vocabulary, tokenizer_config, text_encoder, unet, ddpm = (generation.load_generation_components())

# LOSS FUNCTION UTILIZZATA ANCHE DURANTE IL TRAINING
loss_function = nn.MSELoss()

# SEED PER RENDERE RIPRODUCIBILE LA VALUTAZIONE
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# NUMERO DI SAMPLE UTILIZZATI PER LA CONDITIONING METRIC
NUM_CONDITIONING_SAMPLES = 100

# SELEZIONO I SAMPLE IID E OOD
iid_indexes = splits["test_iid"][:NUM_CONDITIONING_SAMPLES]
ood_indexes = splits["test_ood"][:NUM_CONDITIONING_SAMPLES]

# CREO I DATASET IID E OOD PER LA CONDITIONING METRIC
iid_dataset = CartoonAvatarDataset(original_dataset=original_dataset, split_indexes=iid_indexes,
                                   vocabulary=vocabulary, max_length=tokenizer_config["max_length"])

ood_dataset = CartoonAvatarDataset(original_dataset=original_dataset, split_indexes=ood_indexes,
                                   vocabulary=vocabulary, max_length=tokenizer_config["max_length"])

# CREO I RELATIVI DATA LOADER
iid_loader = create_dataloader(avatar_dataset=iid_dataset, batch_size=20, shuffle=False, num_workers=0)

ood_loader = create_dataloader(avatar_dataset=ood_dataset, batch_size=20, shuffle=False, num_workers=0)

# CALCOLO DEL CONDITIONING MSE GAP
def evaluate_conditioning(data_loader):

    total_correct_mse = 0.0
    total_wrong_mse = 0.0
    total_samples = 0

    text_encoder.eval()
    unet.eval()

    with torch.no_grad():

        for images, token_ids, padding_masks in data_loader:

            images = images.to(generation.device)
            token_ids = token_ids.to(generation.device)
            padding_masks = padding_masks.to(generation.device)

            batch_size = images.size(0)

            # SELEZIONO UN TIMESTEP CASUALE PER OGNI IMMAGINE
            timesteps = torch.randint(0, ddpm.num_timesteps, (batch_size,), device=generation.device, dtype=torch.long)

            # AGGIUNGO RUMORE ALLE IMMAGINI
            noisy_images, noise = ddpm.add_noise(images, timesteps)

            # CREO IL TIME EMBEDDING
            time_embedding = sinusoidal_time_embedding(timesteps)

            # PREVISIONE DEL RUMORE CON LA CAPTION CORRETTA
            correct_text_features = text_encoder(token_ids)

            predicted_noise_correct = unet(noisy_images, time_embedding, correct_text_features, padding_masks)

            correct_mse = loss_function(predicted_noise_correct, noise)

            # CREO DELLE CAPTION SBAGLIATE SPOSTANDO LE CAPTION NEL BATCH
            wrong_token_ids = torch.roll(token_ids, shifts=1, dims=0)

            wrong_padding_masks = torch.roll(padding_masks, shifts=1, dims=0)

            # PREVISIONE DEL RUMORE CON LA CAPTION SBAGLIATA
            wrong_text_features = text_encoder(wrong_token_ids)

            predicted_noise_wrong = unet(noisy_images, time_embedding, wrong_text_features, wrong_padding_masks)

            wrong_mse = loss_function(predicted_noise_wrong, noise)

            total_correct_mse += correct_mse.item() * batch_size
            total_wrong_mse += wrong_mse.item() * batch_size
            total_samples += batch_size

    average_correct_mse = total_correct_mse / total_samples
    average_wrong_mse = total_wrong_mse / total_samples

    conditioning_gap = average_wrong_mse - average_correct_mse

    return average_correct_mse, average_wrong_mse, conditioning_gap

# CALCOLO DELLA CONDITIONING METRIC SUI SAMPLE IID
iid_correct_mse, iid_wrong_mse, iid_conditioning_gap = evaluate_conditioning(iid_loader)

# CALCOLO DELLA CONDITIONING METRIC SUI SAMPLE OOD
ood_correct_mse, ood_wrong_mse, ood_conditioning_gap = evaluate_conditioning(ood_loader)

# STAMPO I RISULTATI IID
print("\nIID RESULTS")
print(f"Correct caption MSE: {iid_correct_mse:.6f}")
print(f"Wrong caption MSE:   {iid_wrong_mse:.6f}")
print(f"Conditioning Gap:    {iid_conditioning_gap:.6f}")

# STAMPO I RISULTATI OOD
print("\nOOD RESULTS")
print(f"Correct caption MSE: {ood_correct_mse:.6f}")
print(f"Wrong caption MSE:   {ood_wrong_mse:.6f}")
print(f"Conditioning Gap:    {ood_conditioning_gap:.6f}")