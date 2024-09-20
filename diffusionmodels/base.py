
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


    @abstractmethod
    def add_noise(self, x, t, noise):
        raise NotImplementedError("add_noise method not implemented")
    

    @abstractmethod
    def loss(self, x, t, noise):
        raise NotImplementedError("loss method not implemented")


    @abstractmethod
    def sample(self, x, t):
        raise NotImplementedError("sample method not implemented")
    
    