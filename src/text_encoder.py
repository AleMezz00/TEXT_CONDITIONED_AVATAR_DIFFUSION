import torch
import torch.nn as nn

#COSTRUZIONE DEL TEXT ENCODER BASATO SU EMBEDDING E TRANSFORMER
class TextEncoder(nn.Module):

    def __init__(self, vocab_size, max_length, hidden_size=64, num_layers=2, num_heads=4):
        super().__init__()

        self.token_embedding_layer = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding_layer = nn.Embedding(max_length, hidden_size)

        # COSTRUZIONE DEL SINGOLO LAYER TRANSFOMER ENCODER
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)

        # DEFINIZIONE DEL TRANSFORMER CON PIU' LAYER TRANSFORMER ENCODER AL SUO INTERNO
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    # FUNZIONE FORWARD PER LA DESCRIZIONE DEL FLUSSO DEI DATI ATTRAVERSO IL TEXT ENCODER
    def forward(self, token_ids):

        sequence_length = token_ids.size(1)

        positions = torch.arange(sequence_length, device=token_ids.device)
        positions = positions.unsqueeze(0).expand(token_ids.size(0), sequence_length)

        token_embeddings = self.token_embedding_layer(token_ids)
        position_embeddings = self.position_embedding_layer(positions)

        text_feature = token_embeddings + position_embeddings

        #GESTIONE DEI TOKEN <PAD> NELLE VARIE SEQUENZE
        padding_mask = token_ids == 0

        text_feature = self.transformer(text_feature, src_key_padding_mask=padding_mask)

        return text_feature
