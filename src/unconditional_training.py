import torch
import torch.nn as nn

import conditional_training as training

from unet import UNet, sinusoidal_time_embedding
from diffusion import DDPM

# DIMENSIONE DELLE TEXT FEATURES UTILIZZATA ANCHE NEL MODELLO CONDITIONAL
TEXT_HIDDEN_SIZE = 64

# CHECKPOINT SEPARATO PER IL MODELLO UNCONDITIONAL
UNCONDITIONAL_CHECKPOINT_PATH = (training.CHECKPOINT_DIR / "unconditional_checkpoint.pt")

# CREAZIONE DI UN CONDITIONING NULLO, SENZA INFORMAZIONI SUL TESTOx
def create_null_conditioning(token_ids):

    batch_size = token_ids.size(0)
    sequence_length = token_ids.size(1)

    # CREO TEXT FEATURES COMPOSTE SOLAMENTE DA ZERI
    text_features = torch.zeros(batch_size, sequence_length, TEXT_HIDDEN_SIZE, device=token_ids.device)

    # NESSUNA POSIZIONE VIENE MASCHERATA
    padding_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=token_ids.device)

    return text_features, padding_mask

# INIZIALIZZAZIONE DEL MODELLO UNCONDITIONAL E DEI COMPONENTI DI TRAINING
def initialize_unconditional_components(learning_rate=1e-4):

    # CREAZIONE DELLA U-NET
    unet = UNet().to(training.device)

    # CREAZIONE DEL PROCESSO DI DIFFUSIONE
    ddpm = DDPM(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device=training.device)

    # LOSS FUNCTION PER CONFRONTARE RUMORE REALE E RUMORE PREVISTO
    loss_function = nn.MSELoss()

    # NELLA BASELINE VIENE ADDESTRATA SOLAMENTE LA U-NET
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

    return unet, ddpm, loss_function, optimizer

# TRAINING DEL MODELLO UNCONDITIONAL PER UNA SINGOLA EPOCH
def train_one_epoch_unconditional(unet, ddpm, train_loader, loss_function, optimizer):

    # IMPOSTO LA U-NET IN MODALITA' TRAINING
    unet.train()

    total_loss = 0.0
    total_samples = 0

    # CICLO FOR PER PROCESSARE UN BATCH ALLA VOLTA PROGRESSIVAMENTE
    for images, token_ids, _ in train_loader:

        images = images.to(training.device)
        token_ids = token_ids.to(training.device)

        batch_size = images.size(0)

        # SELEZIONO UN TIMESTEP CASUALE PER OGNI IMMAGINE
        timesteps = torch.randint(0, ddpm.num_timesteps, (batch_size,), device=training.device, dtype=torch.long)

        # AGGIUNGIAMO, IN BASE AL TIMESTEP CONSIDERATO, IL RUMORE GAUSSIANO CASUALE ALLE IMMAGINI PULITE
        noisy_images, noise = ddpm.add_noise(images, timesteps)

        # AZZERO I GRADIENTI PRIMA DEL NUOVO STEP
        optimizer.zero_grad()

        # CREO UN CONDITIONING NULLO, SENZA INFORMAZIONI SUL TESTO
        text_features, null_padding_mask = create_null_conditioning(token_ids)

        # CREO IL TIME EMBEDDING DEL TIMESTEP
        time_embedding = sinusoidal_time_embedding(timesteps)

        # LA U-NET PREVEDE IL RUMORE SENZA INFORMAZIONI TESTUALI
        predicted_noise = unet(noisy_images, time_embedding, text_features, null_padding_mask)

        # CONFRONTO RUMORE PREVISTO E RUMORE REALE
        loss = loss_function(predicted_noise, noise)

        # BACKPROPAGATION
        loss.backward()

        # AGGIORNAMENTO DEI PARAMETRI DELLA U-NET
        optimizer.step()

        # ACCUMULO LA LOSS PER CALCOLARE LA MEDIA DELL'EPOCA
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    average_train_loss = total_loss / total_samples

    return average_train_loss

# VALIDATION DEL MODELLO UNCONDITIONAL
def validate_unconditional(unet, ddpm, validation_loader, loss_function):

    # IMPOSTO LA U-NET IN MODALITA' EVALUATION
    unet.eval()

    total_loss = 0.0
    total_samples = 0

    # DURANTE LA VALIDATION NON SERVONO I GRADIENTI
    with torch.no_grad():

        # PROCESSO UN BATCH ALLA VOLTA
        for images, token_ids, _ in validation_loader:

            images = images.to(training.device)
            token_ids = token_ids.to(training.device)

            batch_size = images.size(0)

            # SELEZIONO UN TIMESTEP CASUALE PER OGNI IMMAGINE
            timesteps = torch.randint(0, ddpm.num_timesteps, (batch_size,), device=training.device, dtype=torch.long)

            # AGGIUNGO RUMORE ALLE IMMAGINI PULITE
            noisy_images, noise = ddpm.add_noise(images, timesteps)

            # CREO IL CONDITIONING NULLO
            text_features, null_padding_mask = create_null_conditioning(token_ids)

            # CREO IL TIME EMBEDDING
            time_embedding = sinusoidal_time_embedding(timesteps)

            # PREVISIONE DEL RUMORE
            predicted_noise = unet(noisy_images, time_embedding, text_features, null_padding_mask)

            # CALCOLO DELLA LOSS
            loss = loss_function(predicted_noise, noise)

            # ACCUMULO LA LOSS PER LA MEDIA FINALE
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    average_validation_loss = total_loss / total_samples

    return average_validation_loss

# SALVATAGGIO DEL CHECKPOINT DEL MODELLO UNCONDITIONAL
def save_unconditional_checkpoint(unet, optimizer, completed_epoch, best_validation_loss,epochs_without_improvement):

    # CREO LA CARTELLA CHECKPOINT SE NON ESISTE
    training.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # SALVO TUTTE LE INFORMAZIONI NECESSARIE PER RIPRENDERE IL TRAINING
    checkpoint = {
        "epoch": completed_epoch,
        "unet_state_dict": unet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_loss": best_validation_loss,
        "epochs_without_improvement": epochs_without_improvement
    }

    torch.save(checkpoint, UNCONDITIONAL_CHECKPOINT_PATH)

# CARICAMENTO DEL CHECKPOINT DEL MODELLO UNCONDITIONAL
def load_unconditional_checkpoint(unet, optimizer):

    # SE NON ESISTE UN CHECKPOINT, IL TRAINING PARTE DA ZERO
    if not UNCONDITIONAL_CHECKPOINT_PATH.exists():

        return 0, float("inf"), 0

    # CARICO IL CHECKPOINT
    checkpoint = torch.load(UNCONDITIONAL_CHECKPOINT_PATH, map_location=training.device)

    # RIPRISTINO I PARAMETRI DELLA U-NET
    unet.load_state_dict(checkpoint["unet_state_dict"])

    # RIPRISTINO LO STATO DELL'OPTIMIZER
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    completed_epoch = checkpoint["epoch"]
    best_validation_loss = checkpoint["best_validation_loss"]
    epochs_without_improvement = checkpoint["epochs_without_improvement"]

    return completed_epoch, best_validation_loss, epochs_without_improvement

# ADDESTRAMENTO DEL MODELLO UNCONDITIONAL PER PIU' EPOCHE
def train_unconditional_model(unet, ddpm, train_loader, validation_loader, loss_function, optimizer, num_epochs,patience=5):

    # SE ESISTE UN CHECKPOINT, RIPRENDO IL TRAINING DALLO STATO SALVATO
    (completed_epoch, best_validation_loss, epochs_without_improvement) = load_unconditional_checkpoint(unet, optimizer)

    # RIPETO TRAINING E VALIDATION PER IL NUMERO DI EPOCHE RICHIESTO
    for epoch in range(completed_epoch, num_epochs):

        # TRAINING PER UNA SINGOLA EPOCH
        average_train_loss = train_one_epoch_unconditional(unet=unet, ddpm=ddpm, train_loader=train_loader,
                                                           loss_function=loss_function,optimizer=optimizer)

        # VALIDATION DOPO L'EPOCH DI TRAINING
        average_validation_loss = validate_unconditional(unet=unet, ddpm=ddpm, validation_loader=validation_loader,
                                                         loss_function=loss_function)

        # STAMPO I RISULTATI DELL'EPOCH
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Validation Loss: {average_validation_loss:.4f}"
        )

        # EARLY STOPPING
        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        # SALVO LO STATO CORRENTE DEL TRAINING
        save_unconditional_checkpoint(unet=unet, optimizer=optimizer,completed_epoch=epoch + 1,
                                      best_validation_loss=best_validation_loss, epochs_without_improvement=epochs_without_improvement)

        # INTERROMPO IL TRAINING SE NON CI SONO MIGLIORAMENTI
        if epochs_without_improvement >= patience:

            print("Early stopping activated.")
            break

# ESECUZIONE COMPLETA DEL TRAINING DEL MODELLO UNCONDITIONAL
def main():

    # SEED FISSO PER LA RIPRODUCIBILITA'
    training.set_seed(42)

    print(f"Using device: {training.device}")

    # CARICO SPLIT, VOCABOLARIO E CONFIGURAZIONE DEL TOKENIZER
    splits, vocabulary, tokenizer_config, _ = (training.load_saved_configurations())

    # CREO GLI STESSI DATA LOADER UTILIZZATI DAL MODELLO CONDITIONAL
    train_loader, validation_loader = (training.create_training_dataloaders(splits=splits, vocabulary=vocabulary,
                                                                            tokenizer_config=tokenizer_config, batch_size=32, num_workers=0))

    # INIZIALIZZO U-NET, DDPM, LOSS FUNCTION E OPTIMIZER
    unet, ddpm, loss_function, optimizer = (initialize_unconditional_components(learning_rate=1e-4))

    print("Starting unconditional baseline training...")

    # TRAINING DEL MODELLO UNCONDITIONAL
    train_unconditional_model(unet=unet, ddpm=ddpm, train_loader=train_loader, validation_loader=validation_loader, loss_function=loss_function,
                              optimizer=optimizer, num_epochs=50, patience=5)

    print("Unconditional baseline training completed.")

if __name__ == "__main__":
    main()