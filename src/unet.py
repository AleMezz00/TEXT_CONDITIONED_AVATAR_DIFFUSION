import torch
import torch.nn as nn

# CREAZIONE DEI TIME EMBEDDING SINUSOIDALI A PARTIRE DAL TIMESTEP t
def sinusoidal_time_embedding(timesteps, embedding_size=64):

    half_size = embedding_size // 2

    # GENERAZIONE DI FREQUENZE DIFFERENTI UTILIZZATE PER LA RAPPRESENTAZIONE DEL TIMESTEP SU DIVERSE SCALE
    frequencies = 1.0 / (10000 ** (torch.arange(half_size, device=timesteps.device).float() / half_size))

    # COMBINO OGNI TIMESTEP CON LE FREQUENZE CALCOLATE (CALCOLO GLI ANGOLI)
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)

    # CREAZIONE EMBEDDING (PRIMA META' SENO, SECONDA META' COSENO) --> OTTENGO IL VETTORE FINALE DA 64 VALORI
    time_embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    return time_embedding


# CREAZIONE DELLA CLASSE PER IL RESIDUAL BLOCK
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_embedding_size=64):
        super().__init__()

        # PRIMA CONVOLUZIONE DEL BLOCCO
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # NORMALIZZAZIONE DELLE FEATURES
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        # PROIEZIONE DEL TIME EMBEDDING DALLA SUA DIMENSIONE ORIGINALE A QUELLA DI Out_channels
        self.time_projection_layer = nn.Linear(time_embedding_size, out_channels)

        # ACTIVATION FUNCTION
        self.activation = nn.ReLU()

        # SECONDA CONVOLUZIONE DEL BLOCCO
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # SECONDA NORMALIZZAZIONE DA EFFETTUARE DOPO LA SECONDA CONVOLUZIONE
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        # SE IL NUMERO DI CANALI IN INGRESSO E IN USCITA E' DIVERSO, USO UNA CONVOLUZIONE 1x1
        # DEVO GARANTIRE CHE IL NUMERO DI CANALI DI INPUT SIA UGUALE A QUELLO IN USCITA PER EFFETTUARE SUCCESSIVAMENTE LA SOMMA
        if in_channels != out_channels:
            self.residual_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            # SE I CANALI SONO GIA' UGUALI, POSSO LASCIARE IL TENSORE INVARIATO
            self.residual_layer = nn.Identity()

    def forward(self, input_features, time_embedding):

        residual_features = self.residual_layer(input_features)

        # PRIMA CONVOLUZIONE, NORMALIZZAZIONE E ATTIVAZIONE
        processed_features = self.conv1(input_features)
        processed_features = self.norm1(processed_features)
        processed_features = self.activation(processed_features)

        # ADATTAMENTO DIMENSIONI DEL TIME EMBEDDING
        time_features = self.time_projection_layer(time_embedding)

        # AGGIUNGO LE DUE DIMENSIONI SPAZIALI (DA 2 PASSA A 4)
        time_features = time_features.unsqueeze(-1).unsqueeze(-1)

        # AGGIUNGO L'INFORMAZIONE TEMPORALE ALLE FEATURE CHE DEVONO POI ESSERE NUOVAMENTE ELABORATE
        processed_features = processed_features + time_features

        # SECONDA CONVOLUZIONE, NORMALIZZAZIONE E ATTIVAZIONE
        processed_features = self.conv2(processed_features)
        processed_features = self.norm2(processed_features)
        processed_features = self.activation(processed_features)

        # OTTENGO L'OUTPUT DEL RESIDUAL BLOCK SOMMANDO IL RAMO RESIDUO CON I DATI ELABORATI NEL BLOCCO
        output = processed_features + residual_features

        return output


# CREAZIONE DELLA CLASSE DI TEXT CONDITIONING (CROSS-ATTENTION)
class TextConditioningBlock(nn.Module):

    def __init__(self, image_channels=128, text_hidden_size=64, num_heads=4):
        super().__init__()

        # PROIETTA LE FEATURE TESTUALI SULLA DIMENSIONE DI QUELLE VISIVE
        self.text_projection_layer = nn.Linear(text_hidden_size, image_channels)

        # CROSS ATTENTION
        self.cross_attention_layer = nn.MultiheadAttention(embed_dim=image_channels, num_heads=num_heads, batch_first=True)

        # NORMALIZZAZIONE DEL RISULTATO DELLA CROSS ATTENTION
        self.norm_layer = nn.LayerNorm(image_channels)

    def forward(self, image_features, text_features, text_padding_mask=None):

        # SALVO LE DIMENSIONI DELL'IMMAGINE INIZIALE
        batch_size, channels, height, width = image_features.shape

        # TRASFORMO LE FEATURE DELL'IMMAGINE IN UNA SEQUENZA
        image_sequence = image_features.flatten(2).transpose(1, 2)

        # PROIETTA LE FEATURE TESTUALI da 64 a 128 DIMENSIONI
        projected_text_features = self.text_projection_layer(text_features)

        # APPLICO LA CROSS ATTENTION --> IMMAGINE=QUERY, TESTO=KEY e VALUE
        attention_output, _ = self.cross_attention_layer(query=image_sequence, key=projected_text_features, value=projected_text_features,
                                                            key_padding_mask=text_padding_mask, need_weights=False)

        # RESIDUAL CONNECTION TRA FEATURE ORIGINALI E QUELLE APPENA CALCOLATE E APPLICO SU TALE COMBINAZIONE LA NORMALIZZAZIONE
        conditioned_sequence = self.norm_layer(image_sequence + attention_output)

        # RIPORTA LE FEATURE ALLA FORMA ORIGINALE
        conditioned_features = conditioned_sequence.transpose(1, 2).reshape(batch_size, channels, height, width)

        return conditioned_features


# CREAZIONE DELLA CLASSE UNET
class UNet(nn.Module):

    def __init__(self, image_channels=3, base_channels=64, time_embedding_size=64, text_hidden_size=64, conditioning_num_heads=4):
        super().__init__()

        # PRIMA CONVOLUZIONE: TRASFORMO L'IMMAGINE IN UN INSIEME PIU' RICCO DI FEATURE (DA 3 A 64)
        self.initial_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

        # PRIMO RESIDUAL BLOCK ENCODER (LA RISOLUZIONE RIMANE 32x32)
        self.encoder_block1 = ResidualBlock(in_channels=base_channels, out_channels=base_channels, time_embedding_size=time_embedding_size)

        # PRIMO DOWNSAMPLING CHE RIDUCE LA RISOLUZIONE DA 32x32 A 16x16 E AUMENTA I CANALI DA 64 A 128
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)

        # SECONDO RESIDUAL BLOCK ENCODER (IN QUESTO CASO LA RISOLUZIONE E' 16x16)
        self.encoder_block2 = ResidualBlock(in_channels=base_channels * 2, out_channels=base_channels * 2, time_embedding_size=time_embedding_size)

        # TEXT CONDITIONING DELL'ENCODER ALLA RISOLUZIONE 16x16
        self.encoder_conditioning_block = TextConditioningBlock(image_channels=base_channels * 2, text_hidden_size=text_hidden_size,
                                                                            num_heads=conditioning_num_heads)

        # SECONDO DOWNSAMPLING CHE RIDUCE LA RISOLUZIONE DA 16x16 A 8x8 (I CANALI RIMANGONO 128 ANCHE IN USCITA)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)


        # RESIDUAL BLOCK CHE CORRISPONDE AL BLOCCO CENTRALE (QUELLO DI BOTTLENECK) (LA RISOLUZIONE E' 8x8)
        self.bottleneck_block = ResidualBlock(in_channels=base_channels * 2, out_channels=base_channels * 2, time_embedding_size=time_embedding_size)

        # TEXT CONDITIONING NEL BOTTLENECK ALLA RISOLUZIONE 8x8
        self.bottleneck_conditioning_block = TextConditioningBlock(image_channels=base_channels * 2, text_hidden_size=text_hidden_size,
                                                                            num_heads=conditioning_num_heads)


        # PRIMO UPSAMPLING CHE AUMENTA LA RISOLUZIONE DA 8x8 A 16x16 (NUMERO DEI CANALI INVARIATO, SEMPRE 128)
        self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)

        # PRIMO RESIDUAL BLOCK DECODER (RIPRENDE LA PRIMA SKIP CONNECTION)
        self.decoder_block1 = ResidualBlock(in_channels=base_channels * 4, out_channels=base_channels * 2, time_embedding_size=time_embedding_size)

        # TEXT CONDITIONING DEL DECODER ALLA RISOLUZIONE 16x16
        self.decoder_conditioning_block = TextConditioningBlock(image_channels=base_channels * 2, text_hidden_size=text_hidden_size,
                                                                            num_heads=conditioning_num_heads)

        # SECONDO UPSAMPLING CHE AUMENTA LA RISOLUZIONE DA 16x16 A 32x32 E DIMINUISCE IL NUMERO DEI CANALI DA 128 A 64
        self.upsample2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)

        # RESIDUAL BLOCK DOPO LA SECONDA SKIP CONNECTION
        self.decoder_block2 = ResidualBlock(in_channels=base_channels * 2, out_channels=base_channels, time_embedding_size=time_embedding_size)

        # CONVOLUZIONE FINALE CHE RIPORTA IL NUMERO DI CANALI DA 64 A 3 COME ERANO INIZIALMENTE NELL'IMMAGINE
        self.final_conv = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    # FUNZIONE FORWARD DELLA U-NET
    def forward(self, noisy_images, time_embedding, text_features, text_padding_mask=None):

        initial_features = self.initial_conv(noisy_images)

        encoder_features1 = self.encoder_block1(initial_features, time_embedding)

        downsampled_features1 = self.downsample1(encoder_features1)

        encoder_features2 = self.encoder_block2(downsampled_features1, time_embedding)

        # PRIMO TEXT CONDITIONING SULL'ENCODER
        encoder_features2 = self.encoder_conditioning_block(encoder_features2, text_features,text_padding_mask)

        downsampled_features2 = self.downsample2(encoder_features2)


        bottleneck_features = self.bottleneck_block(downsampled_features2, time_embedding)

        # TEXT CONDITIONING SUL BOTTLENECK
        bottleneck_features = self.bottleneck_conditioning_block(bottleneck_features, text_features, text_padding_mask)


        decoder_features1 = self.upsample1(bottleneck_features)

        # AGGIUNGO LA PRIMA SKIP CONNECTION
        decoder_features1 = torch.cat([decoder_features1, encoder_features2], dim=1)

        decoder_features1 = self.decoder_block1(decoder_features1, time_embedding)

        # TEXT CONDITIONING SUL DECODER
        decoder_features1 = self.decoder_conditioning_block(decoder_features1, text_features, text_padding_mask)

        decoder_features2 = self.upsample2(decoder_features1)

        # AGGIUNGO LA SECONDA SKIP CONNECTION
        decoder_features2 = torch.cat([decoder_features2, encoder_features1], dim=1)

        decoder_features2 = self.decoder_block2(decoder_features2, time_embedding)

        # PREDICTION FINALE DEL RUMORE DOPO TUTTO IL PROCESSO DELLA U-NET
        predicted_noise = self.final_conv(decoder_features2)

        return predicted_noise
