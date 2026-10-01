from collections import Counter
from datasets import load_dataset

from preprocessing import DATASET_NAME, DATASET_CONFIG
from attribute_mappings import (
    EYE_COLOR_NAMES,
    GLASSES_NAMES,
    CHIN_LENGTH_NAMES,
    HAIR_COLOR_NAMES,
    EYE_EYEBROW_DISTANCE_NAMES,
    HAIR_NAMES
)

# CARICO IL DATASET
dataset = load_dataset(
    DATASET_NAME,
    DATASET_CONFIG,
    split="train"
)

# CREO I CONTATORI PER LE TIPOLOGIE DI COMBINAZIONI
eye_glasses_counts = Counter()
eye_chin_counts = Counter()
hair_chin_counts = Counter()
eye_eyebrow_distance_counts = Counter()
hair_color_hair_counts = Counter()

# ANALIZZO TUTTI I SAMPLE DEL DATASET
for sample in dataset:

    eye_color = EYE_COLOR_NAMES[sample["eye_color"]]
    glasses = GLASSES_NAMES[sample["glasses"]]
    chin_length = CHIN_LENGTH_NAMES[sample["chin_length"]]
    hair_color = HAIR_COLOR_NAMES[sample["hair_color"]]
    eye_eyebrow_distance = EYE_EYEBROW_DISTANCE_NAMES[sample["eye_eyebrow_distance"]]
    hair = HAIR_NAMES[sample["hair"]]

    eye_glasses_combination = (eye_color, glasses)
    eye_chin_combination = (eye_color, chin_length)
    hair_chin_combination = (hair_color, chin_length)
    eye_eyebrow_distance_combination = (eye_color, eye_eyebrow_distance)
    hair_color_hair_combination = (hair_color, hair)

    eye_glasses_counts[eye_glasses_combination] += 1
    eye_chin_counts[eye_chin_combination] += 1
    hair_chin_counts[hair_chin_combination] += 1
    eye_eyebrow_distance_counts[eye_eyebrow_distance_combination] += 1
    hair_color_hair_counts[hair_color_hair_combination] += 1


# STAMPO LE COMBINAZIONI EYE_COLOR × GLASSES
print("\nEYE COLOR × GLASSES\n")

for combination, count in sorted(eye_glasses_counts.items()):
    print(f"{combination}: {count}")


# STAMPO LE COMBINAZIONI EYE_COLOR × CHIN_LENGTH
print("\nEYE COLOR × CHIN LENGTH\n")

for combination, count in sorted(eye_chin_counts.items()):
    print(f"{combination}: {count}")

# STAMPO LE COMBINAZIONI HAIR_COLOR × CHIN_LENGTH
print("\nHAIR COLOR × CHIN LENGTH\n")

for combination, count in sorted(hair_chin_counts.items()):
    print(f"{combination}: {count}")


# STAMPO LE COMBINAZIONI EYE_COLOR × EYE_EYEBROW_DISTANCE
print("\nEYE COLOR × EYE-EYEBROW DISTANCE\n")

for combination, count in sorted(eye_eyebrow_distance_counts.items()):
    print(f"{combination}: {count}")

# STAMPO LE COMBINAZIONI HAIR_COLOR × HAIR
print("\nHAIR COLOR × HAIR\n")

for combination, count in sorted(hair_color_hair_counts.items()):
    print(f"{combination}: {count}")

# DOPO LA PRECEDENTE ANALISI E' EMERSO CHE LA COMBINAZIONE EYE_COLOR x CHIN_LENGHT E' UNA DELLE PIU' BILANCIATE,
# QUINDI POSSIAMO USARLA PER IL COMPOSITIONAL SPLIT. ESSA INOLTRE RAPPRESENTA UNA COMBINAZIONE COLOR-ARTWORK.
# ALCUNE DI QUESTE COMBINAZIONI VERRRANNO QUINDI ESCLUSE DAL TRAINING PER VERIFICARE SE IL MODELLO RIESCE A GENERARE
# COMBINAZIONI CHE NON HA MAI VISTO PRIMA DI ATTRIBUTI (CHE HA COMUNQUE APPRESO SEPARATAMENTE).