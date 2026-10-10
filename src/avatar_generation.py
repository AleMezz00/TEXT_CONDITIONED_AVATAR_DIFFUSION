import json
import torch

from pathlib import Path
from PIL import Image

from build_tokenizer import encode_caption, pad_sequence
from text_encoder import TextEncoder
from unet import UNet
from diffusion import DDPM

# SELEZIONO IL DEVICE DA UTILIZZARE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DEFINISCO I PERCORSI PRINCIPALI DEL PROGETTO
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_training_checkpoint.pt"

# CARICO LE CONFIGURAZIONI E IL MODELLO CONDITIONAL ADDESTRATO
def load_generation_components():

    # CARICO IL VOCABOLARIO
    with open(DATA_DIR / "vocabulary.json", "r") as file:
        vocabulary = json.load(file)

    # CARICO LA CONFIGURAZIONE DEL TOKENIZER
    with open(DATA_DIR / "tokenizer_config.json", "r") as file:
        tokenizer_config = json.load(file)

    # RICREO IL TEXT ENCODER CON LA STESSA ARCHITETTURA USATA NEL TRAINING
    text_encoder = TextEncoder(vocab_size=len(vocabulary), max_length=tokenizer_config["max_length"], hidden_size=64,
                               num_layers=2, num_heads=4).to(device)

    # RICREO LA U-NET E IL DDPM
    unet = UNet().to(device)
    ddpm = DDPM(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device=device)

    # CARICO IL CHECKPOINT DEL MODELLO CONDITIONAL
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])
    unet.load_state_dict(checkpoint["unet_state_dict"])

    # IMPOSTO I MODELLI IN MODALITA' EVALUATION
    text_encoder.eval()
    unet.eval()

    return vocabulary, tokenizer_config, text_encoder, unet, ddpm

# PREPARAZIONE DEL PROMPT PER IL TEXT ENCODER
def prepare_prompt(prompt, vocabulary, tokenizer_config):

    # CONVERTO IL PROMPT NELLA SEQUENZA DI TOKEN ID
    token_ids = encode_caption(prompt, vocabulary)

    max_length = tokenizer_config["max_length"]

    # SE IL PROMPT E' TROPPO LUNGO LO LIMITO ALLA LUNGHEZZA MASSIMA
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
        token_ids[-1] = vocabulary["<EOS>"]

    # APPLICO IL PADDING FINO ALLA LUNGHEZZA MASSIMA
    token_ids = pad_sequence(token_ids, max_length, vocabulary)

    # CONVERTO LA SEQUENZA IN UN TENSORE E AGGIUNGO LA DIMENSIONE DEL BATCH
    token_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    # CREO LA MASCHERA PER I TOKEN DI PADDING
    padding_mask = token_ids == vocabulary["<PAD>"]

    return token_ids, padding_mask

# GENERAZIONE DELL'AVATAR A PARTIRE DA PROMPT E SEED
def generate_avatar(prompt, seed, vocabulary, tokenizer_config, text_encoder, unet, ddpm):

    # IMPOSTO IL SEED PER RENDERE RIPRODUCIBILE LA GENERAZIONE
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # PREPARO IL PROMPT PER IL TEXT ENCODER
    token_ids, padding_mask = prepare_prompt(prompt, vocabulary, tokenizer_config)

    with torch.no_grad():
        # OTTENGO LE FEATURE TESTUALI DAL TEXT ENCODER
        text_features = text_encoder(token_ids)

    # ESEGUO IL REVERSE SAMPLING DEL DDPM
    generated_image = ddpm.sample(model=unet, text_features=text_features, text_padding_mask=padding_mask,
                                  image_size=32, image_channels=3)

    # RIPORTO I VALORI DELL'IMMAGINE NELL'INTERVALLO [0, 1]
    generated_image = generated_image.clamp(-1.0, 1.0)
    generated_image = (generated_image + 1.0) / 2.0

    # RIMUOVO LA DIMENSIONE DEL BATCH E PORTO I CANALI ALLA FINE
    generated_image = generated_image.squeeze(0)
    generated_image = generated_image.permute(1, 2, 0)

    # CONVERTO IL TENSORE IN UN'IMMAGINE PIL
    generated_image = (generated_image * 255).byte().cpu().numpy()
    generated_image = Image.fromarray(generated_image)

    return generated_image

# ESECUZIONE DELLA GENERAZIONE COMPLETA
def main():

    # CARICO LE CONFIGURAZIONI E I COMPONENTI DEL MODELLO
    vocabulary, tokenizer_config, text_encoder, unet, ddpm = load_generation_components()

    # DEFINISCO LA PARTE FISSA DEL PROMPT
    prompt_prefix = "a cartoon avatar with "

    # MOSTRO LA PARTE FISSA E L'UTENTE INSERISCE SOLO GLI ATTRIBUTI
    print("Insert prompt:")
    print(prompt_prefix, end="")

    user_attributes = input().strip().lower()

    # COSTRUISCO IL PROMPT COMPLETO
    prompt = prompt_prefix + user_attributes

    # CHIEDO ALL'UTENTE IL SEED
    seed_input = input("Insert the seed (press Enter to use 42): ").strip()

    if seed_input == "":
        seed = 42
    else:
        seed = int(seed_input)

    print(f"Using device: {device}")
    print(f"Prompt: {prompt}")
    print(f"Seed: {seed}")

    # GENERO L'AVATAR
    generated_image = generate_avatar(
        prompt=prompt,
        seed=seed,
        vocabulary=vocabulary,
        tokenizer_config=tokenizer_config,
        text_encoder=text_encoder,
        unet=unet,
        ddpm=ddpm
    )

    # SALVO L'IMMAGINE GENERATA
    output_path = PROJECT_ROOT / "generated_avatar.png"
    generated_image.save(output_path)

    print(f"Generated avatar saved in: {output_path}")

if __name__ == "__main__":
    main()