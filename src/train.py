import json
import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from datasets import load_dataset

from preprocessing import DATASET_NAME, DATASET_CONFIG
from dataset import CartoonAvatarDataset, create_dataloader
from text_encoder import TextEncoder
from unet import UNet, sinusoidal_time_embedding
from diffusion import DDPM

# IMPOSTIAMO IL SEED RANDOMICO PER LA RIPRODUCIBILITA'
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# SELEZIONIAMO DOVE DEVE ESSERE EFFETTUATA L'ELABORAZIONE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DEFINISCO IL PERCORSO PRINCIPALE DEL PROGETTO E QUELLO DELLA CARTELLA "DATA"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# CARICO DALLA CARTELLA "DATA" LE CONFIGURAZIONI SALVATE IN PRECEDENZA
def load_saved_configurations():
    with open(DATA_DIR / "splits.json", "r") as file:
        splits = json.load(file)

    with open(DATA_DIR / "vocabulary.json", "r") as file:
        vocabulary = json.load(file)

    with open(DATA_DIR / "tokenizer_config.json", "r") as file:
        tokenizer_config = json.load(file)

    with open(DATA_DIR / "preprocessing_config.json", "r") as file:
        preprocessing_config = json.load(file)

    return splits, vocabulary, tokenizer_config, preprocessing_config

# Create the training and validation datasets and DataLoaders
def create_training_dataloaders(splits, vocabulary, tokenizer_config, batch_size=32, num_workers=0):

    # CARICO IL DATASET ORIGINALE
    original_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")

    # CREO IL DATASET CHE CONTIENE SOLO I SAMPLE PER IL TRAINING
    train_dataset = CartoonAvatarDataset(original_dataset=original_dataset, split_indexes=splits["train"], vocabulary=vocabulary,
                                                                max_length=tokenizer_config["max_length"])

    # CREO IL DATASET CHE CONTIENE SOLO I SAMPLE PER LA VALIDATION
    validation_dataset = CartoonAvatarDataset(original_dataset=original_dataset, split_indexes=splits["validation"],
                                                vocabulary=vocabulary, max_length=tokenizer_config["max_length"])

    # CREO IL DATA LOADER PER IL TRAINING E EFFETTUO LO SHUFFLE DEI SAMPLE PER L'ADDESTRAMENTO
    train_loader = create_dataloader(avatar_dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # CREO IL DATA LOADER PER LA VALIDATION SENZA LA NECESSITA' DELLO SHUFFLE
    validation_loader = create_dataloader(avatar_dataset=validation_dataset, batch_size=batch_size, shuffle=False,
                                                                        num_workers=num_workers)

    return train_loader, validation_loader

# INIZIALIZZAZIONE DEL MODELLO E DELLE PRINCIPALI COMPONENTI DI TRAINING
def initialize_training_components(vocabulary, tokenizer_config, learning_rate=1e-4):

    # CREAZIONE TEXT ENCODER
    text_encoder = TextEncoder(vocab_size=len(vocabulary), max_length=tokenizer_config["max_length"], hidden_size=64,
                                                            num_layers=2, num_heads=4).to(device)

    # CREAZIONE U-NET
    unet = UNet().to(device)

    # CREAZIONE DDPM
    ddpm = DDPM(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device=device)

    # LOSS FUNCTION
    loss_function = nn.MSELoss()

    # PARAMETRI DI TEXT ENCODER E U-NET CHE DOVRANNO ESSERE OTTIMIZZATI
    trainable_parameters = (list(text_encoder.parameters()) + list(unet.parameters()))

    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)

    return text_encoder, unet, ddpm, loss_function, optimizer

# ADDESTRAMENTO DEL MODELLO PER UNA EPOCH (ELABORA UNA VOLTA TUTTI I BATCH DEL TRAINING SET)
def train_one_epoch(text_encoder, unet, ddpm, train_loader, loss_function, optimizer):

    # IMPOSTO LA MODALITA' DI TRAINING PER I VALORI DEI NOSTRI MODELLI
    text_encoder.train()
    unet.train()

    total_loss = 0.0

    # CICLO FOR PER PROCESSARE UN BATCH ALLA VOLTA PROGRESSIVAMENTE
    for images, token_ids, padding_masks in train_loader:

        images = images.to(device)
        token_ids = token_ids.to(device)
        padding_masks = padding_masks.to(device)

        batch_size = images.size(0)

        # SELEZIONIAMO CASUALEMNTE UN TIMESTEP PER CIASCUNA DELLE IMMAGINI CHE COSTITUISCE IL BATCH
        timesteps = torch.randint(0, ddpm.num_timesteps, (batch_size,),  device=device, dtype=torch.long)

        # AGGIUNGIAMO, IN BASE AL TIMESTEP CONSIDERATO, IL RUMORE GAUSSIANO CASUALE ALLE IMMAGINI PULITE
        noisy_images, noise = ddpm.add_noise(images, timesteps)


