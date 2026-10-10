import conditional_training as conditional_training
import unconditional_training as unconditional_training

# CONFIGURAZIONE DELLO SMOKE TEST UNCONDITIONAL
SMOKE_TRAIN_SAMPLES = 16
SMOKE_VALIDATION_SAMPLES = 8
SMOKE_BATCH_SIZE = 4
SMOKE_EPOCHS = 30

# DEFINISCO I CHECKPOINT SEPARATI PER LO SMOKE TEST UNCONDITIONAL
SMOKE_CHECKPOINT_PATH = (conditional_training.CHECKPOINT_DIR /"unconditional_smoke_test_checkpoint_64.pt")
SMOKE_BEST_CHECKPOINT_PATH = (conditional_training.CHECKPOINT_DIR /"best_unconditional_smoke_test_checkpoint_64.pt")

unconditional_training.BEST_UNCONDITIONAL_CHECKPOINT_PATH = SMOKE_BEST_CHECKPOINT_PATH

# ASSEGNO I CHECKPOINT DELLO SMOKE TEST AL MODULO DI TRAINING
unconditional_training.UNCONDITIONAL_CHECKPOINT_PATH = SMOKE_CHECKPOINT_PATH

# CREAZIONE DEI DATA LOADER RIDOTTI PER LO SMOKE TEST
def create_smoke_dataloaders(splits, vocabulary, tokenizer_config):

    smoke_splits = {"train": splits["train"][:SMOKE_TRAIN_SAMPLES], "validation": splits["validation"][:SMOKE_VALIDATION_SAMPLES]}

    train_loader, validation_loader = (conditional_training.create_training_dataloaders( splits=smoke_splits, vocabulary=vocabulary,
                                                                                         tokenizer_config=tokenizer_config, batch_size=SMOKE_BATCH_SIZE,
                                                                                         num_workers=0))

    return train_loader, validation_loader


# ESECUZIONE DELLO SMOKE TEST UNCONDITIONAL
def main():

    # SEED FISSO PER LA RIPRODUCIBILITA'
    conditional_training.set_seed(42)

    print(
        f"Unconditional smoke test device: "
        f"{conditional_training.device}"
    )

    # ELIMINO EVENTUALI CHECKPOINT PRECEDENTI DELLO SMOKE TEST
    if SMOKE_CHECKPOINT_PATH.exists():
        SMOKE_CHECKPOINT_PATH.unlink()

    if SMOKE_BEST_CHECKPOINT_PATH.exists():
        SMOKE_BEST_CHECKPOINT_PATH.unlink()

    # CARICO LE CONFIGURAZIONI SALVATE
    splits, vocabulary, tokenizer_config, _ = (conditional_training.load_saved_configurations())

    # CREO I DATA LOADER RIDOTTI
    train_loader, validation_loader = (create_smoke_dataloaders( splits=splits,vocabulary=vocabulary, tokenizer_config=tokenizer_config))

    # INIZIALIZZO LA BASELINE UNCONDITIONAL
    unet, ddpm, loss_function, optimizer = (unconditional_training.initialize_unconditional_components(learning_rate=1e-4))

    print("Starting unconditional smoke test...")

    # TRAINING MOLTO BREVE
    unconditional_training.train_unconditional_model(
        unet=unet,
        ddpm=ddpm,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        num_epochs=SMOKE_EPOCHS,
        patience=SMOKE_EPOCHS
    )

    print("Unconditional smoke test completed.")

if __name__ == "__main__":
    main()