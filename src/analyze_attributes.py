# SCRIPT UTILE PER EFFETTUARE UN CONTROLLO SUGLI ELEMENTI PRESENTI ALL'INTERNO DEL DATASET

from datasets import load_dataset
from collections import Counter
from PIL import Image
from io import BytesIO
import os

# CARICO IL DATASET CON GLI ATTRIBUTI
dataset = load_dataset("cgarciae/cartoonset", "10k+features")
samples = dataset["train"]

# ATTRIBUTI CHE VOGLIO ANALIZZARE
attributes = [
    "chin_length",
    "eye_color",
    "hair_color",
    "eyebrow_width",
    "eye_lashes",
    "eye_eyebrow_distance",
    "glasses",
    "facial_hair",
    "hair"
]

# CONTO QUANTE VOLTE COMPARE OGNI VALORE
for attribute in attributes:
    counts = Counter(samples[attribute])

    print("\n", attribute)
    print(counts)

# CREO UNA CARTELLA PER ANALIZZARE VISIVAMENTE I COLORI DEGLI OCCHI
os.makedirs("../data/eye_color_samples", exist_ok=True)

# SALVO 3 AVATAR PER OGNI VALORE DI EYE_COLOR
for color_value in range(5):
    saved_images = 0

    for index, sample in enumerate(samples):
        if sample["eye_color"] == color_value and sample["glasses"] == 11:
            image = Image.open(BytesIO(sample["img_bytes"])).convert("RGB")

            image.save(
                f"../data/eye_color_samples/"
                f"eye_color_{color_value}_sample_{saved_images}.png"
            )

            saved_images += 1

            if saved_images == 3:
                break

# CREO UNA CARTELLA PER ANALIZZARE VISIVAMENTE I COLORI DEI CAPELLI
os.makedirs("../data/hair_color_samples", exist_ok=True)

# SALVO 3 AVATAR PER OGNI VALORE DI HAIR_COLOR
for color_value in range(10):
    saved_images = 0

    for index, sample in enumerate(samples):
        if sample["hair_color"] == color_value:
            image = Image.open(BytesIO(sample["img_bytes"])).convert("RGB")

            image.save(
                f"../data/hair_color_samples/"
                f"hair_color_{color_value}_sample_{saved_images}.png"
            )

            saved_images += 1

            if saved_images == 3:
                break

# CREO UNA CARTELLA PER ANALIZZARE VISIVAMENTE I DIVERSI TIPI DI CIGLIA
os.makedirs("../data/eye_lashes_samples", exist_ok=True)

# SALVO 3 AVATAR SENZA OCCHIALI PER OGNI VALORE DI EYE_LASHES
for eye_lashes_value in range(2):
    saved_images = 0

    for index, sample in enumerate(samples):
        if (
            sample["eye_lashes"] == eye_lashes_value
            and sample["glasses"] == 11
        ):
            image = Image.open(BytesIO(sample["img_bytes"])).convert("RGB")

            image.save(
                f"../data/eye_lashes_samples/"
                f"eye_lashes_{eye_lashes_value}_sample_{saved_images}.png"
            )

            saved_images += 1

            if saved_images == 3:
                break

# CREO UNA CARTELLA PER ANALIZZARE VISIVAMENTE LA DISTANZA TRA OCCHI E SOPRACCIGLIA
os.makedirs("../data/eye_eyebrow_distance_samples", exist_ok=True)

# SALVO 3 AVATAR SENZA OCCHIALI PER OGNI VALORE DI EYE_EYEBROW_DISTANCE
for distance_value in range(3):
    saved_images = 0

    for index, sample in enumerate(samples):
        if (
            sample["eye_eyebrow_distance"] == distance_value
            and sample["glasses"] == 11
        ):
            image = Image.open(BytesIO(sample["img_bytes"])).convert("RGB")

            image.save(
                f"../data/eye_eyebrow_distance_samples/"
                f"eye_eyebrow_distance_{distance_value}_sample_{saved_images}.png"
            )

            saved_images += 1

            if saved_images == 3:
                break

# CREO UNA CARTELLA PER ANALIZZARE VISIVAMENTE LE DIVERSE LUNGHEZZE DEL MENTO
os.makedirs("../data/chin_length_controlled_samples", exist_ok=True)

# SALVO 3 AVATAR PER OGNI CHIN_LENGTH MANTENENDO LA STESSA FACE_SHAPE ED ESCLUDENDO BARBA E BAFFI
for chin_length_value in range(3):
    saved_images = 0

    for index, sample in enumerate(samples):
        if (
            sample["chin_length"] == chin_length_value
            and sample["face_shape"] == 1
            and sample["facial_hair"] == 14
        ):
            image = Image.open(BytesIO(sample["img_bytes"])).convert("RGB")

            image.save(
                f"../data/chin_length_controlled_samples/"
                f"chin_length_{chin_length_value}_sample_{saved_images}.png"
            )

            saved_images += 1

            if saved_images == 3:
                break

# CREO UNA CARTELLA PER ANALIZZARE VISIVAMENTE LE DIVERSE LARGHEZZE DELLE SOPRACCIGLIA
os.makedirs("../data/eyebrow_width_controlled_samples", exist_ok=True)

# SALVO 3 AVATAR PER OGNI VALORE DI EYEBROW_WIDTH MANTENENDO LO STESSO TIPO DI SOPRACCIGLIA E SENZA OCCHIALI
for eyebrow_width_value in range(3):
    saved_images = 0

    for index, sample in enumerate(samples):
        if (
            sample["eyebrow_width"] == eyebrow_width_value
            and sample["eyebrow_shape"] == 0
            and sample["glasses"] == 11
        ):
            image = Image.open(BytesIO(sample["img_bytes"])).convert("RGB")

            image.save(
                f"../data/eyebrow_width_controlled_samples/"
                f"eyebrow_width_{eyebrow_width_value}_sample_{saved_images}.png"
            )

            saved_images += 1

            if saved_images == 3:
                break

# CREO UNA CARTELLA PER ANALIZZARE VISIVAMENTE I DIVERSI TIPI DI OCCHIALI
os.makedirs("../data/glasses_samples", exist_ok=True)

# SALVO 2 AVATAR PER OGNI VALORE DI GLASSES
for glasses_value in range(12):
    saved_images = 0

    for index, sample in enumerate(samples):
        if sample["glasses"] == glasses_value:
            image = Image.open(BytesIO(sample["img_bytes"])).convert("RGB")

            image.save(
                f"../data/glasses_samples/"
                f"glasses_{glasses_value}_sample_{saved_images}.png"
            )

            saved_images += 1

            if saved_images == 2:
                break

# CREO UNA CARTELLA PER ANALIZZARE VISIVAMENTE I DIVERSI TIPI DI BARBA E BAFFI
os.makedirs("../data/facial_hair_samples", exist_ok=True)

# SALVO 2 AVATAR PER OGNI VALORE DI FACIAL_HAIR
for facial_hair_value in range(15):
    saved_images = 0

    for index, sample in enumerate(samples):
        if sample["facial_hair"] == facial_hair_value:
            image = Image.open(BytesIO(sample["img_bytes"])).convert("RGB")

            image.save(
                f"../data/facial_hair_samples/"
                f"facial_hair_{facial_hair_value}_sample_{saved_images}.png"
            )

            saved_images += 1

            if saved_images == 2:
                break

# CREO UNA CARTELLA PER ANALIZZARE VISIVAMENTE I DIVERSI TIPI DI CAPELLI
os.makedirs("../data/hair_samples", exist_ok=True)

# SALVO 1 AVATAR PER OGNI VALORE DI HAIR
for hair_value in range(111):
    for index, sample in enumerate(samples):
        if sample["hair"] == hair_value:
            image = Image.open(BytesIO(sample["img_bytes"])).convert("RGB")

            image.save(
                f"../data/hair_samples/"
                f"hair_{hair_value}.png"
            )

            break