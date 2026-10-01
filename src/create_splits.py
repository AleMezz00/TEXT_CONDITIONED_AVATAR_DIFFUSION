from datasets import load_dataset
from preprocessing import DATASET_NAME, DATASET_CONFIG, PREPROCESSING_CONFIG
from attribute_mappings import EYE_COLOR_NAMES, CHIN_LENGTH_NAMES

from sklearn.model_selection import train_test_split

import json
from pathlib import Path

# COMBINAZIONI ESCLUSE DAL TRAINING PER IL COMPOSITIONAL-OOD (EYE_COLOR x CHIN_LENGTH)
OOD_COMBINATIONS = {
    ("black", "long"),
    ("blue", "medium"),
    ("green", "short")
}

# CARICO IL DATASET
dataset = load_dataset(
    DATASET_NAME,
    DATASET_CONFIG,
    split="train"                                               # l'intero dataset è definito come train e comprende i 10k sample
)

# CREO LE LISTE CHE CONTERRANNO GLI INDICI DEI SAMPLE
ood_indexes = []
remaining_indexes = []

# SEPARO I SAMPLE OOD DA TUTTI GLI ALTRI
for index, sample in enumerate(dataset):

    eye_color = EYE_COLOR_NAMES[sample["eye_color"]]
    chin_length = CHIN_LENGTH_NAMES[sample["chin_length"]]

    combination = (eye_color, chin_length)

    if combination in OOD_COMBINATIONS:
        ood_indexes.append(index)
    else:
        remaining_indexes.append(index)

# DIVIDO I SAMPLE RIMANENTI IN TRAINING, VALIDATION E TEST IID USANDO train_test_split
# IN PARTICOLARE temp_indices E' SOLO TEMPORANEO PERCHE' A SUA VOLTA DOVRA' ESSERE SUDDIVISO PER VALIDATION E TEST
train_indexes, temp_indexes = train_test_split(remaining_indexes, test_size=0.20,random_state=42)

# DIVIDO I SAMPLE TEMPORANEI IN VALIDATION E TEST IID
validation_indexes, test_indexes = train_test_split(temp_indexes, test_size=0.50, random_state=42)

# EFFETTUO ORA UN CONTROLLO SUL NUMERO TOTALE DI SAMPLE PER CIASCUNO SPLIT DOPO LA DIVISIONE
print(f"Total samples: {len(dataset)}")
print(f"Train samples: {len(train_indexes)}")
print(f"Validation samples: {len(validation_indexes)}")
print(f"Test IID samples: {len(test_indexes)}")
print(f"Compositional-OOD samples: {len(ood_indexes)}")

# CONTROLLO LE COMBINAZIONI PRESENTI NEL TRAINING SET DEGLI ATTRIBUTI SPECIFICATI E VERIFICO CHE NESSUNA DI ESSE CORRISPONDA
# AD UNA DELLE COMBINAZIONI RISERVATE ALL'OOD
train_combinations = set()

for index in train_indexes:
    sample = dataset[index]

    eye_color = EYE_COLOR_NAMES[sample["eye_color"]]
    chin_length = CHIN_LENGTH_NAMES[sample["chin_length"]]

    combination = (eye_color, chin_length)
    train_combinations.add(combination)

ood_found = False

for combination in train_combinations:
    if combination in OOD_COMBINATIONS:
        print(f"ERROR: OOD combination {combination} found in training")
        ood_found = True

if not ood_found:
    print("No OOD combinations are present in the training set")

# EFFETTUO UN SECONDO CONTROLLO: VERIFICO CHE I SINGOLI ATTRIBUTI OOD SIANO PRESENTI NEL TRAINING
train_eye_colors = set()
train_chin_lengths = set()

for index in train_indexes:
    sample = dataset[index]

    eye_color = EYE_COLOR_NAMES[sample["eye_color"]]
    chin_length = CHIN_LENGTH_NAMES[sample["chin_length"]]

    train_eye_colors.add(eye_color)
    train_chin_lengths.add(chin_length)


print(f"Eye colors present in training: {train_eye_colors}")
print(f"Chin lengths present in training: {train_chin_lengths}")

# ORA CREO UN DIZIONARIO PYTHON IN CUI SALVO I VARI INDICI CORRISPONDENTI A CIASCUNO SPLIT
splits = {
    "train": train_indexes,
    "validation": validation_indexes,
    "test_iid": test_indexes,
    "test_ood": ood_indexes
}

# SALVO TUTTI GLI INDICI DEI DIVERSI SET IN UN FILE JSON CHE PUO' ESSERE RIUTILIZZATO
# INOLTRE SALVO ANCHE IL FILE JSON PER LA CONFIGURAZIONE DEL PREPROCESSING
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_PATH = PROJECT_ROOT / "data" / "splits.json"
PREPROCESSING_CONFIG_PATH = PROJECT_ROOT / "data" / "preprocessing_config.json"

with open(SPLITS_PATH, "w") as file:
    json.dump(splits, file, indent=4)

print(f"Splits saved in: {SPLITS_PATH}")

with open(PREPROCESSING_CONFIG_PATH, "w") as file:
    json.dump(PREPROCESSING_CONFIG, file, indent=4)

print(f"Preprocessing configuration saved in: {PREPROCESSING_CONFIG_PATH}")