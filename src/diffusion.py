import torch
from unet import sinusoidal_time_embedding
from preprocessing import IMAGE_SIZE

# CLASSE PER IL DDPM
class DDPM:

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):

        self.num_timesteps = num_timesteps
        self.device = device

        # CREO IL NOISE SCHEDULE LINEARE
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)

        # CALCOLO ALPHA PER OGNI TIMESTEP
        self.alphas = 1.0 - self.betas

        # CALCOLO ALPHA BAR, OVVERO IL PRODOTTO CUMULATIVO DEGLI ALPHA FINO A QUELLO SPECIFICO TIMESTEP
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    # FUNZIONE PER L'AGGIUNTA DEL RUMORE AD UN'IMMAGINE
    def add_noise(self, clean_images, timesteps, noise=None):

        # GENERAZIONE RUMORE CASUALE SE QUESTO NON VIENE FORNITO INIZIALMENTE
        if noise is None:
            noise = torch.randn_like(clean_images)

        # SELEZIONE E MODIFICA DI ALPHA BAR PER OGNI TIMESTEP
        alpha_bar_t = self.alpha_bars[timesteps]
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)

        # APPLICAZIONE FORWARD NOISING DEL DDPM
        noisy_images = (torch.sqrt(alpha_bar_t) * clean_images + torch.sqrt(1.0 - alpha_bar_t) * noise)

        return noisy_images, noise

    # FUNZIONE DI RIMOZIONE DEL RUMORE (SINGOLO STEP)
    def remove_noise_step(self, noisy_images, predicted_noise, timesteps):

        # RECUPERO I COEFFICIENTI RELATIVI A QUESTO TIMESTEP
        beta_t = self.betas[timesteps].view(-1, 1, 1, 1)
        alpha_t = self.alphas[timesteps].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bars[timesteps].view(-1, 1, 1, 1)

        # CALCOLO REVERSE STEP
        previous_images = (1.0 / torch.sqrt(alpha_t)) * (noisy_images - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise)

        # GENERO COMPONENTE CASUALE DEL REVERSE STEP
        random_noise = torch.randn_like(noisy_images)

        # DEFINIZIONE MASCHERA PER L'AGGIUNTA DEL RUMORE CASUALE (AD OGNI STEP TRANNE t = 0)
        nonzero_mask = (timesteps != 0).float().view(-1, 1, 1, 1)

        previous_images = (previous_images + nonzero_mask * torch.sqrt(beta_t) * random_noise)

        return previous_images

    # FUNZIONE PER IL SAMPLING LOOP COMPLETO
    def sample(self, model, text_features, text_padding_mask=None, image_size=IMAGE_SIZE, image_channels=3):

        batch_size = text_features.size(0)

        current_images = torch.randn(batch_size, image_channels, image_size, image_size, device=self.device)

        # EVITO CHE VENGANO CALCOLATI I GRADIENTI IN QUESTO CASO
        with torch.no_grad():

            for timestep in reversed(range(self.num_timesteps)):

                timesteps = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)

                # CALCOLO DEL TIME EMBEDDING A PARTIRE DAL TIMESTEP CORRENTE
                time_embedding = sinusoidal_time_embedding(timesteps)

                # STIMA DEL RUMORE DA PARTE DELLA U-NET (IL NOSTRO MODEL)
                predicted_noise = model(current_images, time_embedding, text_features, text_padding_mask)

                # CALCOLO DELL'IMMAGINE AL TIMESTEP PRECEDENTE (RIMOZIONE GRADUALE DEL RUMORE)
                current_images = self.remove_noise_step(current_images, predicted_noise, timesteps)

                # MANTENGO I VALORI NELL'INTERVALLO ATTESO DELLE IMMAGINI NORMALIZZATE
                current_images = current_images.clamp(-1.0, 1.0)

        return current_images
