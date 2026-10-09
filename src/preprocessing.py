from datasets import load_dataset
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms

from attribute_mappings import (
    EYE_COLOR_NAMES,
    HAIR_COLOR_NAMES,
    HAIR_NAMES,
    EYE_LASHES_NAMES,
    EYE_EYEBROW_DISTANCE_NAMES,
    CHIN_LENGTH_NAMES,
    EYEBROW_WIDTH_NAMES,
    GLASSES_NAMES,
    FACIAL_HAIR_NAMES
)

# DATI UTILI PER LA CONFIGURAZIONE DEL PREPROCESSING
IMAGE_SIZE = 64
DATASET_NAME = "cgarciae/cartoonset"
DATASET_CONFIG = "10k+features"

# INSERISCO IN UN UNICO DIZIONARIO LE PRINCIPALI IMPOSTAZIONI DI CONFIGURAZIONE DEL PREPROCESSING
PREPROCESSING_CONFIG = {
    "dataset_name": DATASET_NAME,
    "dataset_config": DATASET_CONFIG,
    "image_size": IMAGE_SIZE,
    "image_mode": "RGB",
    "normalization_mean": [0.5, 0.5, 0.5],
    "normalization_std": [0.5, 0.5, 0.5],
    "caption_type": "deterministic",
    "caption_attributes": [
        "eye_color",
        "hair_color",
        "hair",
        "glasses",
        "facial_hair",
        "eye_lashes",
        "eye_eyebrow_distance",
        "eyebrow_width",
        "chin_length"
    ]
}

# TRASFORMAZIONE E NORMALIZZAZIONE DELLE IMMAGINI
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CON QUESTA FUNZIONE EFFETTUO IL RESIZING DEGLI AVATAR ALLA RISOLUZIONE SPECIFICATA (da 500x500 a 64x64) E APPLICO LA NORMALIZZAZIONE
def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image = image_transform(image)

    return image

# GENERAZIONE DELLA CAPTION DETERMINISTICA A PARTIRE DAGLI ATTRIBUTI DELL'AVATAR. ESSA VIENE USATA PER PREPARARE I DATI ED
# ADDESTRARE IL MODELLO TEXT-CONDITIONED
def generate_caption(sample):

    eye_color = EYE_COLOR_NAMES[sample["eye_color"]]
    hair_color = HAIR_COLOR_NAMES[sample["hair_color"]]
    eye_lashes = EYE_LASHES_NAMES[sample["eye_lashes"]]
    eye_eyebrow_distance = EYE_EYEBROW_DISTANCE_NAMES[sample["eye_eyebrow_distance"]]
    chin_length = CHIN_LENGTH_NAMES[sample["chin_length"]]
    eyebrow_width = EYEBROW_WIDTH_NAMES[sample["eyebrow_width"]]
    glasses = GLASSES_NAMES[sample["glasses"]]
    facial_hair = FACIAL_HAIR_NAMES[sample["facial_hair"]]
    hair = HAIR_NAMES[sample["hair"]]

    if hair == "bald":
        hair_description = "bald"
    else:
        hair_description = f"{hair_color} {hair}"

    caption = (
        f"a cartoon avatar with {eye_color} eyes, "
        f"{hair_description}, "
        f"{glasses}, "
        f"{facial_hair}, "
        f"{eye_lashes}, "
        f"{eye_eyebrow_distance} eye-eyebrow distance, "
        f"{eyebrow_width} eyebrows, "
        f"and a {chin_length} chin. "
    )

    return caption

