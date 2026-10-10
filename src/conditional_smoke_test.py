import conditional_training as training

# CONFIGURAZIONE SCELTA PER LO SMOKE TEST
SMOKE_TRAIN_SAMPLES = 16
SMOKE_VALIDATION_SAMPLES = 8
SMOKE_BATCH_SIZE = 4
SMOKE_EPOCHS = 30

# DEFINIAMO DEI CHECKPOINT SEPARATI PER LO SMOKE TEST
SMOKE_CHECKPOINT_PATH = training.CHECKPOINT_DIR / "smoke_test_checkpoint.pt"
SMOKE_BEST_CHECKPOINT_PATH = training.CHECKPOINT_DIR / "best_smoke_test_checkpoint.pt"

training.CHECKPOINT_PATH = SMOKE_CHECKPOINT_PATH
training.BEST_CHECKPOINT_PATH = SMOKE_BEST_CHECKPOINT_PATH

# CREO DEI DATA LOADERS PER IL TRAINING E LA VALIDATION DEL NOSTRO SMOKE TEST
def create_smoke_dataloaders(splits, vocabulary, tokenizer_config):

    smoke_splits = {"train": splits["train"][:SMOKE_TRAIN_SAMPLES],
                    "validation": splits["validation"][:SMOKE_VALIDATION_SAMPLES]}

    # RIUSIAMO LA STESSA FUNZIONE USATA IN conditional_training.py PER LA CREAZIONE DEI DATA LOADER
    train_loader, validation_loader = training.create_training_dataloaders(splits=smoke_splits, vocabulary=vocabulary,
                                                tokenizer_config=tokenizer_config, batch_size=SMOKE_BATCH_SIZE, num_workers=0)

    return train_loader, validation_loader

# ESECUZIONE DELLO SMOKE TEST
def main():

    training.set_seed(42)

    print(f"Smoke test device: {training.device}")

    # ELIMINO EVENTUALI CHECKPOINT PRECEDENTI DELLO SMOKE TEST
    if SMOKE_CHECKPOINT_PATH.exists():
        SMOKE_CHECKPOINT_PATH.unlink()

    if SMOKE_BEST_CHECKPOINT_PATH.exists():
        SMOKE_BEST_CHECKPOINT_PATH.unlink()

    splits, vocabulary, tokenizer_config, _ = (training.load_saved_configurations())

    # CREO DEI DATA LOADERS APPOSITI
    train_loader, validation_loader = create_smoke_dataloaders(splits=splits, vocabulary=vocabulary, tokenizer_config=tokenizer_config)

    # INIZIALIZZO GLI STESSI MODELLI UTILIZZATI ANCHE NELLA FASE DI TRAINING DEL MODELLO
    text_encoder, unet, ddpm, loss_function, optimizer = (training.initialize_training_components(vocabulary=vocabulary,
                                                                            tokenizer_config=tokenizer_config,learning_rate=1e-4))

    print("Starting smoke test...")

    # ESECUZIONE TRAINING PER LO SMOKE TEST
    training.train_model(
        text_encoder=text_encoder,
        unet=unet,
        ddpm=ddpm,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        num_epochs=SMOKE_EPOCHS,
        patience=SMOKE_EPOCHS
    )

    print("Smoke test completed.")

if __name__ == "__main__":
    main()

