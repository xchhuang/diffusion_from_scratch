
import torch
import numpy as np
import sys
sys.path.append('../')
from abc import ABC, abstractmethod


class BaseDiffusionModel(ABC):

    def __init__(self):
        """
        Initializes the diffusion model.
        Args:
            :
        """
        super(BaseDiffusionModel, self).__init__()

        self.num_train_timesteps = 1000
        self.num_eval_timesteps = 250
        self.skip = int(self.num_train_timesteps // self.num_eval_timesteps)
        self.guidance_scale = 1.0
        self.scaled_vae_latent_factor = 0.18215

        # for ddpm/ddim
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.beta_schedule = 'linear'
        if self.beta_schedule == 'linear':
            self.betas = np.linspace(self.beta_start, self.beta_end, self.num_train_timesteps).astype(np.float32)
        else:
            raise NotImplementedError(f"beta_schedule={self.beta_schedule} not implemented")


    @abstractmethod
    def add_noise(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Func:
            Adds noise to the input data x as the forward process of diffusion models.
        Args:
            x (torch.Tensor, latents, [batch, frames, channels, height, width]): The input data.
            t (torch.Tensor, 0-num_steps, [batch]): The current timestep or timesteps as a tensor.
            noise (torch.Tensor, gaussian, [batch, frames, channels, height, width]): The noise tensor to be added to the input data.
        Returns:
            torch.Tensor: The noisy data (torch.Tensor).
        """
        raise NotImplementedError("add_noise method not implemented")


    @abstractmethod
    def loss(self, x, t, noise):
        raise NotImplementedError("loss method not implemented")


    @abstractmethod
    def sample(self, x, t):
        raise NotImplementedError("sample method not implemented")
    

    
    def set_neuralnet(self, neuralnet):
        self.neuralnet = neuralnet
    