import json
import torch

from io import BytesIO
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from torchmetrics.image.fid import FrechetInceptionDistance

import avatar_generation as generation

from preprocessing import (DATASET_NAME, DATASET_CONFIG, IMAGE_SIZE, generate_caption)

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
original_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")

# CARICO IL MODELLO CONDITIONAL GIA' ADDESTRATO
vocabulary, tokenizer_config, text_encoder, unet, ddpm = (generation.load_generation_components())

# NUMERO DI SAMPLE UTILIZZATI PER IL CALCOLO DELLA FID
NUM_FID_SAMPLES = 50

# SELEZIONO I SAMPLE DAL TEST IID
fid_indexes = splits["test_iid"][:NUM_FID_SAMPLES]

# INIZIALIZZO LA METRICA FID
fid_metric = FrechetInceptionDistance(feature=2048).to(generation.device)

# AGGIUNGO ALLA FID LE IMMAGINI REALI DEL TEST IID
for index in fid_indexes:
    sample = original_dataset[index]

    real_image = Image.open(BytesIO(sample["img_bytes"])).convert("RGB")

    real_image = real_image.resize((IMAGE_SIZE, IMAGE_SIZE))

    real_tensor = pil_to_tensor(real_image).unsqueeze(0)
    real_tensor = real_tensor.to(generation.device)

    fid_metric.update(real_tensor, real=True)

# AGGIUNGO ALLA FID LE IMMAGINI GENERATE DAL MODELLO CONDITIONAL
for sample_number, index in enumerate(fid_indexes):

    sample = original_dataset[index]
    prompt = generate_caption(sample)

    generated_image = generation.generate_avatar(
        prompt=prompt,
        seed=42 + sample_number,
        vocabulary=vocabulary,
        tokenizer_config=tokenizer_config,
        text_encoder=text_encoder,
        unet=unet,
        ddpm=ddpm
    )

    generated_tensor = pil_to_tensor(generated_image).unsqueeze(0)
    generated_tensor = generated_tensor.to(generation.device)

    fid_metric.update(generated_tensor, real=False)

# CALCOLO IL VALORE FINALE DELLA FID
fid_value = fid_metric.compute()

print(f"FID: {fid_value.item():.4f}")