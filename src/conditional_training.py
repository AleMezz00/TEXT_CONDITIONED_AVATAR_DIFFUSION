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

# DEFINISCO I PERCORSI E I CHECKPOINT DEL TRAINING
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "training_checkpoint_64.pt"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_training_checkpoint_64.pt"

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

# CREAZIONE DEI DATASET E DEI DATA LOADER PER TRAINING E VALIDATION
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
    total_samples = 0

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

        # AZZERO I GRADIENTI DELL'OPTIMIZER PER CIASCUNO STEP (PYTORCH LI ACCUMULEREBBE AUTOMATICAMENTE)
        optimizer.zero_grad()

        # ELABORAZIONE DEI TOKEN DELLE CAPTION
        text_features = text_encoder(token_ids)

        # OTTENIAMO I TIME EMBEDDING A PARTIRE DAI TIMESTEP SELEZIONATI
        time_embedding = sinusoidal_time_embedding(timesteps)

        # TRAMITE LA U-NET SI EFFETTUA LA PREVISIONE DEL RUMORE
        predicted_noise = unet(noisy_images, time_embedding, text_features, padding_masks)

        # CON LA LOSS FUNCTION CONFRONTIAMO IL RUMORE PREVISTO CON QUELLO REALE
        loss = loss_function(predicted_noise, noise)

        # CALCOLIAMO I GRADIENTI CON LA BACKPROPAGATION
        loss.backward()

        # CONSEGUENTE AGGIORNAMENTO DEI PARAMETRI DI TEXT ENCODER E U-NET
        optimizer.step()

        # INCREMENTIAMO LA LOSS
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    # CALCOLO LA LOSS MEDIA DELL'INTERA EPOCA
    average_train_loss = total_loss / total_samples

    return average_train_loss

# EVALUATION DEL MODELLO SU SAMPLE DIFFERENTI DA QUELLI DEL TRAINING
def validate(text_encoder, unet, ddpm, validation_loader, loss_function):

    # IMPOSTO LA MODALITA' DI EVALUATION PER I VALORI DEI NOSTRI MODELLI
    text_encoder.eval()
    unet.eval()

    total_loss = 0.0
    total_samples = 0

    # DISATTIVO IL CALCOLO DEI GRADIENTI IN QUESTO PROCESSO (NON NECESSARI)
    with torch.no_grad():

        for images, token_ids, padding_masks in validation_loader:

            images = images.to(device)
            token_ids = token_ids.to(device)
            padding_masks = padding_masks.to(device)

            batch_size = images.size(0)

            timesteps = torch.randint(0, ddpm.num_timesteps, (batch_size,), device=device, dtype=torch.long)

            noisy_images, noise = ddpm.add_noise(images, timesteps)

            text_features = text_encoder(token_ids)

            time_embedding = sinusoidal_time_embedding(timesteps)

            predicted_noise = unet(noisy_images, time_embedding, text_features, padding_masks)

            loss = loss_function(predicted_noise, noise)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    average_validation_loss = total_loss / total_samples

    return average_validation_loss

# SALVATAGGIO DI UN CHECKPOINT NEL PERCORSO SPECIFICATO
def save_checkpoint(text_encoder, unet, optimizer, completed_epoch, best_validation_loss,
                    epochs_without_improvement, checkpoint_path):

    # CREO LA CARTELLA DEI CHECKPOINT SE NON ESISTE
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": completed_epoch,
        "text_encoder_state_dict": text_encoder.state_dict(),
        "unet_state_dict": unet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_loss": best_validation_loss,
        "epochs_without_improvement": epochs_without_improvement
    }

    # SALVATAGGIO DEL CHECKPOINT NELLA CARTELLA CREATA APPOSTIAMENTE IN PRECEDENZA
    torch.save(checkpoint, checkpoint_path)

# CARICAMENTO DEL CHECKPOINT CORRENTE PER RIPRENDERE IL TRAINING
def load_checkpoint(text_encoder, unet, optimizer):

    # SE IL CHECKPOINT NON ESISTE RIPARTO DALL'EPOCA 0, CON UNA LOSS PARI A INFINITO E NESSUNA EPOCA SENZA MIGLIORAMENTO
    if not CHECKPOINT_PATH.exists():
        return 0, float("inf"), 0

    # CARICO IL CHECKPOINT DEFINITO NEL FILE APPOSITO
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    # RECUPERO I PARAMETRI SALVATI DEI MODELLI E I DATI DELL'OPTIMIZER
    text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])
    unet.load_state_dict(checkpoint["unet_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    completed_epoch = checkpoint["epoch"]
    best_validation_loss = checkpoint["best_validation_loss"]
    epochs_without_improvement = checkpoint["epochs_without_improvement"]

    return completed_epoch, best_validation_loss, epochs_without_improvement

# ADDESTRAMENTO DEL MODELLO PER UN DETERMINATO NUMERO DI EPOCHE
def train_model(text_encoder, unet, ddpm, train_loader, validation_loader, loss_function, optimizer, num_epochs, patience=5):

    # SE C'E' UN CHECKPOINT RECUPERO QUEI DATI E RIPARTO DA LI'
    (completed_epoch, best_validation_loss, epochs_without_improvement) = load_checkpoint(text_encoder, unet, optimizer)

    for epoch in range(completed_epoch, num_epochs):

        # EFFETTUO IL TRAINING PER UNA SINGOLA EPOCH
        average_train_loss = train_one_epoch(text_encoder, unet, ddpm, train_loader, loss_function, optimizer)

        # EVALUATION DEL MODELLO IN BASE AI DATI OTTENUTI
        average_validation_loss = validate(text_encoder, unet, ddpm, validation_loader, loss_function)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Validation Loss: {average_validation_loss:.4f}"
        )

        # EARLY STOPPING - CONTROLLO SE C'E' STATO UN MIGLIORAMENTO NELLA VALIDATION
        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            epochs_without_improvement = 0

            # SALVO ANCHE IL MODELLO MIGLIORE
            save_checkpoint(text_encoder, unet, optimizer, completed_epoch=epoch + 1, best_validation_loss=best_validation_loss,
                            epochs_without_improvement=epochs_without_improvement, checkpoint_path=BEST_CHECKPOINT_PATH)

        else:
            epochs_without_improvement += 1

        # SALVO IL CHECKPOINT CORRENTE PER POTER RIPRENDERE IL TRAINING
        save_checkpoint(text_encoder, unet, optimizer, completed_epoch=epoch + 1,best_validation_loss=best_validation_loss,
                        epochs_without_improvement=epochs_without_improvement, checkpoint_path=CHECKPOINT_PATH)

        # INTERROMPO IL TRAINING SE NON SI HA UN MIGLIORAMENTO PER UN DETERMINATO NUMERO DI EPOCHE CONSECUTIVE
        if epochs_without_improvement >= patience:
            print("Early stopping activated.")
            break

# ESECUZIONE DELLA PIPELINE COMPLETA DEL TRAINING
def main():

    # SEED PER LA RIPRODUCIBILITA'
    set_seed(42)

    print(f"Using device: {device}")

    # CARICO LE CONFIGURAZIONI COMPLETE SALVATE IN PRECEDENZA
    splits, vocabulary, tokenizer_config, preprocessing_config = (load_saved_configurations())

    # CREO I DATA LOADER DI TRAINING E VALIDATION
    train_loader, validation_loader = create_training_dataloaders(splits=splits, vocabulary=vocabulary,
                                          tokenizer_config=tokenizer_config, batch_size=32, num_workers=0)

    # INIZIALIZZO IL MODELLO E I COMPONENTI
    text_encoder, unet, ddpm, loss_function, optimizer = (initialize_training_components(vocabulary=vocabulary,
                                                            tokenizer_config=tokenizer_config, learning_rate=1e-4))

    # TRAINING
    train_model(
        text_encoder=text_encoder,
        unet=unet,
        ddpm=ddpm,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        num_epochs=50,
        patience=5
    )

if __name__ == "__main__":
    main()