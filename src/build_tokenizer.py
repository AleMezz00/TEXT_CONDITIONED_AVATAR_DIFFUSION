import json
from pathlib import Path
from datasets import load_dataset

from preprocessing import (DATASET_NAME, DATASET_CONFIG, generate_caption)

# CREO LA FUNZIONE DI TOKENIZZAZIONE DELLE CAPTION
def tokenize(caption):
    caption = caption.lower()
    caption = caption.replace(",", "")
    caption = caption.replace(".", "")
    tokens = caption.split()

    return tokens

# FUNZIONE PER LA CONVERSIONE DI UNA CAPTION NELLA CORRISPONDENTE SEQUENZA DI TOKEN ID
def encode_caption(caption, vocabulary):
    tokens = tokenize(caption)

    token_ids = [vocabulary["<BOS>"]]

    for token in tokens:
        token_id = vocabulary.get(token, vocabulary["<UNK>"])
        token_ids.append(token_id)

    token_ids.append(vocabulary["<EOS>"])

    return token_ids

# DEFINIZIONE DEI TOKEN SPECIALI DEL VOCABOLARIO
SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]

# APPLICAZIONE DEL PADDING PER UNIFORMARE LA LUNGHEZZA DELLE SEQUENZE
def pad_sequence(token_ids, max_length, vocabulary):
    pad_id = vocabulary["<PAD>"]

    while len(token_ids) < max_length:
        token_ids.append(pad_id)

    return token_ids

# DEFINISCO IL PERCORSO DI SPLITS.JSON CHE CONTIENE GLI INDICI ASSEGNATI AI DIVERSI SPLIT DEL DATASET
# SUBITO DOPO RIPRENDO IL FILE IN MODALITA' LETTURA
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_PATH = PROJECT_ROOT / "data" / "splits.json"

with open(SPLITS_PATH, "r") as file:
    splits = json.load(file)

# PRENDO SOLO LA LISTA ASSOCIATA ALLA CHIAVE TRAIN (COSì CONSIDERO SOLO IL TRAIN SET)
train_indexes = splits["train"]

# CARICO IL DATASET (TRAIN IN QUESTO CASO E' LO SPLIT INTERO ORIGINALE DA 10K CHIAMATO DA CartoonSet)
dataset = load_dataset(
    DATASET_NAME,
    DATASET_CONFIG,
    split="train"
)

# VADO A SALVARE IN UN ARRAY TUTTE LE CAPTION CHE GENERIAMO RISPETTO ACIASCUN SAMPLE PRESENTE NEL TRAINING SET
train_captions = []

for index in train_indexes:
    sample = dataset[index]
    caption = generate_caption(sample)
    train_captions.append(caption)

# ORA TRAMITE LA FUNZIONE TOKENIZE RACCOLGO TUTTI I TOKEN PRESENTI NELLE CAPTION DI TRAINING
training_tokens = set()

for caption in train_captions:
    tokens = tokenize(caption)

    for token in tokens:
        training_tokens.add(token)

# COSTRUISCO CONCRETAMENTE IL VOCABOLARIO CON TUTTI I TOKEN INDIVIDUATI E I TOKEN SPECIALI AGGIUNTI IN PRECEDENZA
sorted_training_tokens = sorted(training_tokens)

vocabulary = {}

for token in SPECIAL_TOKENS:
    vocabulary[token] = len(vocabulary)

for token in sorted_training_tokens:
    vocabulary[token] = len(vocabulary)

# CALCOLO DELLA LUNGHEZZA MASSIMA DELLE CAPTION DI TRAINING
max_caption_length = 0

for caption in train_captions:
    tokens = tokenize(caption)
    caption_length = len(tokens) + 2  # +2 per <BOS> e <EOS>

    if caption_length > max_caption_length:
        max_caption_length = caption_length

# ORA DETERMINIAMO IL PERCORSO DEL VOCABOLARIO DEFINITO E LO SALVIAMO
VOCABULARY_PATH = PROJECT_ROOT / "data" / "vocabulary.json"

with open(VOCABULARY_PATH, "w") as file:
    json.dump(vocabulary, file, indent=4)

print(f"Vocabulary saved in: {VOCABULARY_PATH}")

# DEFINIZIONE DELLA CONFIGURAZIONE DEL TOKENIZER
tokenizer_config = {
    "vocabulary_size": len(vocabulary),
    "max_length": max_caption_length,
    "pad_token": "<PAD>",
    "unk_token": "<UNK>",
    "bos_token": "<BOS>",
    "eos_token": "<EOS>"
}

TOKENIZER_CONFIG_PATH = PROJECT_ROOT / "data" / "tokenizer_config.json"

# SALVATAGGIO DELLA CONFIGURAZIONE DEL TOKENIZER
with open(TOKENIZER_CONFIG_PATH, "w") as file:
    json.dump(tokenizer_config, file, indent=4)

print(f"Tokenizer configuration saved in: {TOKENIZER_CONFIG_PATH}")

# CONTROLLO FINALE DEL VOCABOLARIO E DEL TOKENIZER
print(f"Training captions: {len(train_captions)}")
print(f"Vocabulary size: {len(vocabulary)}")
print(f"Maximum sequence length: {max_caption_length}")

print(f"PAD token ID: {vocabulary['<PAD>']}")
print(f"UNK token ID: {vocabulary['<UNK>']}")
print(f"BOS token ID: {vocabulary['<BOS>']}")
print(f"EOS token ID: {vocabulary['<EOS>']}")