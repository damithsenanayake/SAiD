"""Define the diffusion models which are used as SAiD model
"""
from abc import abstractmethod, ABC
import inspect
from typing import Dict, List, Optional, Union
from diffusers import DDPMScheduler, SchedulerMixin
import numpy as np
import torch
from torch import nn
import torchaudio
from transformers import (
    ProcessorMixin,
    PretrainedConfig,
    PreTrainedModel,
    Wav2Vec2Config,
    Wav2Vec2Model,
    Wav2Vec2Processor,
)
from .unet_1d_condition import UNet1DConditionModel
from .vae import BCVAE


class SAID(ABC, nn.Module):
    """Abstract class of SAiD models"""

    audio_encoder: nn.Module
    audio_processor: ProcessorMixin
    sampling_rate: int
    denoiser: nn.Module
    noise_scheduler: SchedulerMixin

    def process_audio(
        self, waveform: Union[np.ndarray, torch.Tensor, List[np.ndarray]]
    ) -> torch.FloatTensor:
        """Process the waveform to fit the audio encoder

        Parameters
        ----------
        waveform : Union[np.ndarray, torch.Tensor, List[np.ndarray]]
            - np.ndarray, torch.Tensor: (audio_seq_len,)
            - List[np.ndarray]: each (audio_seq_len,)

        Returns
        -------
        torch.FloatTensor
            (Batch_size, T_a), Processed mono waveform
        """
        out = self.audio_processor(
            waveform, sampling_rate=self.sampling_rate, return_tensors="pt"
        )["input_values"]
        return out

    @abstractmethod
    def get_audio_embedding(self, waveform: torch.FloatTensor) -> torch.FloatTensor:
        """Return the audio embedding of the waveform

        Parameters
        ----------
        waveform : torch.FloatTensor
            (Batch_size, T_a), Processed mono waveform

        Returns
        -------
        torch.FloatTensor
            (Batch_size, embed_seq_len, embed_size), Generated audio embedding
        """
        pass

    def get_random_timesteps(self, batch_size: int) -> torch.LongTensor:
        """Return the random timesteps

        Parameters
        ----------
        batch_size : int
            Size of the batch

        Returns
        -------
        torch.LongTensor
            (batch_size,), random timesteps
        """
        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps,
            (batch_size,),
            dtype=torch.long,
        )
        return timesteps

    def add_noise(
        self, samples: torch.FloatTensor, timesteps: torch.LongTensor
    ) -> Dict[str, torch.FloatTensor]:
        """Add the noise into the sample

        Parameters
        ----------
        samples : torch.FloatTensor
            Samples to be noised
        timesteps : torch.LongTensor
            (num_timesteps,), Timestep of the noise scheduler

        Returns
        -------
        Dict[str, torch.FloatTensor]
            {
                "noisy_samples": Noisy samples
                "noise": Added noise
            }
        """
        noise = torch.randn(samples.shape, device=samples.device)
        noisy_samples = self.noise_scheduler.add_noise(samples, noise, timesteps)

        output = {
            "noisy_samples": noisy_samples,
            "noise": noise,
        }
        return output


class SAID_UNet1D(SAID):
    """SAiD model implemented using U-Net 1D model"""

    def __init__(
        self,
        audio_config: Optional[Wav2Vec2Config] = None,
        audio_processor: Optional[Wav2Vec2Processor] = None,
        noise_scheduler: Optional[SchedulerMixin] = None,
        vae_x_dim: int = 32,
        vae_h_dim: int = 16,
        vae_z_dim: int = 8,
        diffusion_steps: int = 1000,
        latent_scale: float = 0.18215,
    ):
        """Constructor of SAID_Wav2Vec2

        Parameters
        ----------
        audio_config : Optional[Wav2Vec2Config], optional
            Wav2Vec2Config object, by default None
        audio_processor : Optional[Wav2Vec2Processor], optional
            Wav2Vec2Processor object, by default None
        noise_scheduler : Optional[SchedulerMixin], optional
            scheduler object, by default None
        x_dim : int
            Dimension of the input, by default 32
        h_dim : int
            Dimension of the hidden layer, by default 16
        z_dim : int
            Dimension of the latent, by default 8
        diffusion_steps : int
            The number of diffusion steps, by default 50
        latent_scale : float
            Scaling the latent, by default 0.18215
        """
        super(SAID_Wav2Vec2, self).__init__()

        # Audio-related
        self.audio_config = (
            audio_config if audio_config is not None else Wav2Vec2Config()
        )
        self.audio_encoder = Wav2Vec2Model(self.audio_config)
        self.audio_processor = (
            audio_processor
            if audio_processor is not None
            else Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        )
        self.sampling_rate = self.audio_processor.feature_extractor.sampling_rate

        # VAE
        self.vae = BCVAE(x_dim=vae_x_dim, h_dim=vae_h_dim, z_dim=vae_z_dim)
        self.latent_scale = latent_scale

        # Denoiser-related
        self.denoiser = UNet1DConditionModel(
            in_channels=vae_z_dim,
            out_channels=vae_z_dim,
            cross_attention_dim=self.audio_config.hidden_size,
        )
        self.noise_scheduler = (
            noise_scheduler
            if noise_scheduler is not None
            else DDPMScheduler(
                num_train_timesteps=diffusion_steps,
                beta_start=1e-4,
                beta_end=2e-2,
                beta_schedule="squaredcos_cap_v2",
            )
        )

    def forward(
        self,
        noisy_samples: torch.FloatTensor,
        timesteps: torch.LongTensor,
        audio_embedding: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Return the predicted noise in the noisy samples

        Parameters
        ----------
        noisy_samples : torch.FloatTensor
            (Batch_size, coeffs_seq_len, z_dim), Sequence of noisy coefficient latents
        timesteps : torch.LongTensor
            (Batch_size,) or (1,), Timesteps
        audio_embedding : torch.FloatTensor
            (Batch_size, embedding_seq_len, embedding_size), Sequence of audio embeddings

        Returns
        -------
        torch.FloatTensor
            (Batch_size, coeffs_seq_len, num_coeffs), Sequence of predicted noises
        """
        noise_pred = self.denoiser(noisy_samples, timesteps, audio_embedding)
        return noise_pred

    def get_audio_embedding(self, waveform: torch.FloatTensor) -> torch.FloatTensor:
        features = self.audio_encoder(waveform).last_hidden_state
        return features

    def get_latent(
        self,
        coeffs: torch.FloatTensor,
        use_noise: bool = True,
        align_noise: bool = True,
        do_scaling: bool = True,
    ) -> torch.FloatTensor:
        """Convert blendshape coefficients into the latents

        Parameters
        ----------
        coeffs : torch.FloatTensor
            (Batch_size, sample_seq_len, x_dim), Blendshape coefficients
        use_noise : bool, optional
            Whether using noises when reconstructing the coefficients, by default True
        align_noise : bool, optional
            Whether the noises are the same in each batch, by default True
        do_scaling : bool, optional
            Whether scaling the latent, by default True

        Returns
        -------
        torch.FloatTensor
            (Batch_size, sample_seq_len, z_dim), Latents of the coefficients
        """
        latent_stats = self.vae.encode(coeffs)
        latent = (
            self.vae.reparametrize(
                latent_stats["mean"], latent_stats["log_var"], align_noise
            )
            if use_noise
            else mean
        )
        if do_scaling:
            latent *= self.latent_scale
        return latent

    def inference(
        self,
        waveform_processed: torch.FloatTensor,
        init_samples: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 0.0,
        eta: float = 0.0,
        fps: int = 60,
        save_intermediate: bool = False,
    ) -> Dict[str, Union[torch.FloatTensor, List[torch.FloatTensor]]]:
        """Inference pipeline

        Parameters
        ----------
        waveform_processed : torch.FloatTensor
            (Batch_size, T_a), Processed mono waveform
        init_samples : Optional[torch.FloatTensor], optional
            (Batch_size, sample_seq_len, x_dim), Starting point for the process, by default None
        num_inference_steps : int, optional
            The number of denoising steps, by default 100
        guidance_scale : float, optional
            Guidance scale in classifier-free guidance, by default 7.5
        eta : float, optional
            Eta in DDIM, by default 0.0
        fps : int, optional
            The number of frames per second, by default 60

        Returns
        -------
        Dict[str, Union[torch.FloatTensor, List[torch.FloatTensor]]]
            {
                "Result": torch.FloatTensor, (Batch_size, sample_seq_len, x_dim), Generated blendshape coefficients
                "Intermediate": List[torch.FloatTensor], (Batch_size, sample_seq_len, x_dim), Intermediate blendshape coefficients
            }
        """
        batch_size = waveform_processed.shape[0]
        waveform_len = waveform_processed.shape[1]
        in_channels = self.denoiser.in_channels
        device = waveform_processed.device
        do_classifier_free_guidance = guidance_scale > 1.0
        window_size = int(waveform_len / self.sampling_rate * fps)

        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)

        if init_samples is None:
            latents = torch.randn(batch_size, window_size, in_channels, device=device)
        else:
            latent_stats = self.vae.encode(init_samples)
            latents = self.vae.reparametrize(
                latent_stats["mean"], latent_stats["log_var"], True
            )
            # Todo: Adding additional noise would be necessary

        # Scaling the latent
        latents *= self.latent_scale * self.noise_scheduler.init_noise_sigma

        audio_embedding = self.get_audio_embedding(waveform_processed)
        if do_classifier_free_guidance:
            uncond_waveform = [np.zeros((waveform_len)) for _ in range(batch_size)]
            uncond_waveform_processed = self.process_audio(uncond_waveform).to(device)
            uncond_audio_embedding = self.get_audio_embedding(uncond_waveform_processed)

            audio_embedding = torch.cat([uncond_audio_embedding, audio_embedding])

        # Prepare extra kwargs for the scheduler step
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.noise_scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = eta

        intermediates = []

        for t in self.noise_scheduler.timesteps:
            if save_intermediate:
                interm = self.vae.decode(latents / self.latent_scale)
                intermediates.append(interm)

            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.noise_scheduler.scale_model_input(
                latent_model_input, t
            )

            noise_pred = self.forward(latent_model_input, t, audio_embedding)

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_audio - noise_pred_uncond
                )

            latents = self.noise_scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        # Re-scaling the latent
        latents /= self.latent_scale
        result = self.vae.decode(latents)

        output = {
            "Result": result,
            "Intermediate": intermediates,
        }

        return output
