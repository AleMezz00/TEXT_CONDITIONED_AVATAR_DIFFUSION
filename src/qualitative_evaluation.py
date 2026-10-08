import json
from pathlib import Path
from datasets import load_dataset

import avatar_generation as generation

from preprocessing import DATASET_NAME, DATASET_CONFIG, generate_caption

# DEFINISCO I PERCORSI PRINCIPALI
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_PATH = DATA_DIR / "splits.json"

# CARTELLA IN CUI SALVERO' I RISULTATI DELLA VALUTAZIONE
RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CARICO GLI SPLIT SALVATI
with open(SPLITS_PATH, "r") as file:
    splits = json.load(file)

# CARICO IL DATASET ORIGINALE
original_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")

# CARICO IL MODELLO CONDITIONAL GIA' ADDESTRATO
vocabulary, tokenizer_config, text_encoder, unet, ddpm = generation.load_generation_components()

# INDICO IL NUMERO DI SAMPLE DA RIPRENDERE PER CIASCUNO SPLIT
NUM_SAMPLES = 3
SEED = 42

# CREO I PERCORSI E LE RELATIVE CARTELLE PER SALVARE LE GENERAZIONI IID E OOD
IID_RESULTS_DIR = RESULTS_DIR / "iid"
OOD_RESULTS_DIR = RESULTS_DIR / "ood"

IID_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OOD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# GENERO ALCUNI AVATAR DAL TEST IID
for sample_number, index in enumerate(splits["test_iid"][:NUM_SAMPLES]):

    sample = original_dataset[index]
    prompt = generate_caption(sample)

    generated_image = generation.generate_avatar(
        prompt=prompt,
        seed=SEED,
        vocabulary=vocabulary,
        tokenizer_config=tokenizer_config,
        text_encoder=text_encoder,
        unet=unet,
        ddpm=ddpm
    )

    output_path = IID_RESULTS_DIR / f"iid_{sample_number}.png"
    generated_image.save(output_path)

    print(f"IID {sample_number} | {prompt}")


# GENERO ALCUNI AVATAR DAL TEST COMPOSITIONAL-OOD
for sample_number, index in enumerate(splits["test_ood"][:NUM_SAMPLES]):

    sample = original_dataset[index]
    prompt = generate_caption(sample)

    generated_image = generation.generate_avatar(
        prompt=prompt,
        seed=SEED,
        vocabulary=vocabulary,
        tokenizer_config=tokenizer_config,
        text_encoder=text_encoder,
        unet=unet,
        ddpm=ddpm
    )

    output_path = OOD_RESULTS_DIR / f"ood_{sample_number}.png"
    generated_image.save(output_path)

    print(f"OOD {sample_number} | {prompt}")
