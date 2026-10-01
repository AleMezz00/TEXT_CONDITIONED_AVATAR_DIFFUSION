import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing import preprocess_image, generate_caption
from build_tokenizer import encode_caption, pad_sequence

# CREO UN MIO DATASET PERSONALIZZATO COMPATIBILE CON IL SISTEMA PYTORCH
class CartoonAvatarDataset(Dataset):
    def __init__(self, original_dataset, split_indexes, vocabulary, max_length=3):

        self.original_dataset = original_dataset
        self.split_indexes = split_indexes
        self.vocabulary = vocabulary
        self.max_length = max_length

    # FUNZIONE CHE DETERMINA IL NUMERO DI CAMPIONI PRESENTI ALL'INTERNO DEL DATASET (RELATIVA ALLO SPLIT SUL QUALE LAVORIAMO)
    def __len__(self):
        return len(self.split_indexes)

    # FUNZIONE CHE PERMETTE DI RECUPERARE UN SINGOLO CAMPIONE ED EFFETTUA SU DI ESSO IL PREPROCESSING
    def __getitem__(self, index):
        # RECUPERO L'INDICE DEL SAMPLE NEL DATASET ORIGINALE E RIPRENDO TALE SAMPLE
        original_index = self.split_indexes[index]
        sample = self.original_dataset[original_index]

        # GENERO IL TENSORE DELL'IMMAGINE DEL SAMPLE E LA CAPTION TESTUALE
        image = preprocess_image(sample["img_bytes"])
        caption = generate_caption(sample)

        # TRASFORMO LA CAPTION IN TOKEN ID, APPLICO  IL PADDING E TRASFORMO IL TUTTO IN UN TENSORE PYTORCH
        token_ids = encode_caption(caption, self.vocabulary)
        token_ids = pad_sequence(token_ids, self.max_length, self.vocabulary)
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        # PADDING MASK CHE ASSOCIA TRUE AI TOKEN DI PADDING
        padding_mask = token_ids == self.vocabulary["<PAD>"]

        return image, token_ids, padding_mask

# Create a DataLoader for a PyTorch Dataset
def create_dataloader(avatar_dataset, batch_size=32, shuffle=False, num_workers=0):

    dataloader = DataLoader(avatar_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader

