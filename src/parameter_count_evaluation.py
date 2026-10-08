import avatar_generation as generation

# CARICO IL MODELLO CONDITIONAL GIA' ADDESTRATO
vocabulary, tokenizer_config, text_encoder, unet, ddpm = (generation.load_generation_components())

# CALCOLO IL NUMERO DI PARAMETRI DEL TEXT ENCODER
text_encoder_parameters = sum(parameter.numel() for parameter in text_encoder.parameters())

# CALCOLO IL NUMERO DI PARAMETRI DELLA U-NET
unet_parameters = sum(parameter.numel() for parameter in unet.parameters())

# CALCOLO IL NUMERO TOTALE DI PARAMETRI DEL MODELLO CONDITIONAL
total_parameters = text_encoder_parameters + unet_parameters

# STAMPO I RISULTATI
print(f"Text Encoder parameters: {text_encoder_parameters:,}")
print(f"U-Net parameters:        {unet_parameters:,}")
print(f"Total parameters:        {total_parameters:,}")