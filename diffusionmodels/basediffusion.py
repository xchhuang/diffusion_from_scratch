
import torch
import numpy as np
import sys
sys.path.append('../')
from abc import ABC, abstractmethod



def logit_normal_sampling(device, batch_size, m=0.0, s=1.0):
    """
    Logit-Normal sampling for diffusion timesteps.
    
    Args:
        batch_size (int): Number of samples to generate.
        m (float): Location parameter (mean of the normal distribution in logit space).
        s (float): Scale parameter (stddev of the normal distribution in logit space).
        device (str): Device for computation ("cuda" or "cpu").
        
    Returns:
        torch.Tensor: Samples from the logit-normal distribution in the range (0, 1).
    """
    # Step 1: Sample from the normal distribution N(m, s)
    u = torch.normal(mean=m, std=s, size=(batch_size,), device=device)
    
    # Step 2: Map to the unit interval using the sigmoid function
    t = torch.sigmoid(u)
    
    return t



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

        self.timesteps_sample_scheduler = 'logit_normal'  # linear, logit_normal
        
    
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
    def sample_noise(self, data):
        # noise can be customized/sampled in a different way, such as BNDM, Logit-Normal Sampling, SNR
        raise NotImplementedError("sample_noise method not implemented")
    

    @abstractmethod
    def sample(self, x, t):
        raise NotImplementedError("sample method not implemented")
    

    
    def set_neuralnet(self, neuralnet):
        self.neuralnet = neuralnet
    

    # timesteps can be customized/sampled in a different way, such as Logit-Normal Sampling, SNR
    def sample_timesteps(self, device, batch_size):
        t = torch.randint(0, self.num_train_timesteps, (batch_size, )).to(device)
        if self.timesteps_sample_scheduler == 'linear':
            pass
        elif self.timesteps_sample_scheduler == 'logit_normal':
            t = logit_normal_sampling(device, batch_size)
            t = (t * self.num_train_timesteps).long()
        else:
            raise NotImplementedError(f"timesteps_sample_scheduler={self.timesteps_sample_scheduler} not implemented")
        return t
    

    
        
    


        