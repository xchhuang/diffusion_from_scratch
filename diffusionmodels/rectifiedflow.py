
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('../')
from diffusionmodels.base import BaseDiffusionModel


class RectifiedFlow(BaseDiffusionModel):
    def __init__(self):
        super(RectifiedFlow, self).__init__()
        print("===> Using RectifiedFlow")
        pass

    def add_noise(self, x, t, noise):
        alpha = t / self.num_train_timesteps
        if x.ndim == 4:
            alpha = alpha[:, None, None, None]
        elif x.ndim == 5:
            alpha = alpha[:, None, None, None, None]
        else:
            raise ValueError("x should have 4 or 5 dimensions")
        x_alpha = (1 - alpha) * x + (alpha) * noise
        return x_alpha
    

    def loss(self, pred, tar):
        return F.mse_loss(pred, tar)


    def sample(self, model, x_start, cond=None):
        
        x_alpha = x_start
        start_step = 0
        skip = self.skip
        num_train_timesteps = self.num_train_timesteps
        device = x_start.device
        seq = list(range(start_step, num_train_timesteps, skip))
        seq = [s + skip for s in seq]
        seq_next = [0] + list(seq[:-1])

        use_reverse = True
        if use_reverse:
            seq = list(reversed(seq))
            seq_next = list(reversed(seq_next))

        for t, t_next in zip(seq, seq_next):

            tt = torch.tensor(t).to(device)
            tt_next = torch.tensor(t_next).to(device)

            alpha_start = tt / num_train_timesteps
            alpha_end = tt_next / num_train_timesteps
            
            # cfg
            # d = model.forward_cfg(torch.cat([x_alpha, x_alpha], 0), alpha_start, emb, cfg_scale=args.cfg_scale)
            # latent_model_input  = torch.cat([x_alpha] * 2)
            # with torch.no_grad():
            # noise_pred = model(latent_model_input, t, encoder_hidden_states=emb).sample
            # noise_pred = model(latent_model_input, tt)

            with torch.no_grad():
                if cond is None:
                    noise_pred = model(x_alpha, tt.unsqueeze(0))
                else:
                    x_alpha_inp = torch.cat([x_alpha, x_alpha], 0)
                    tt_inp = torch.cat([tt.unsqueeze(0), tt.unsqueeze(0)], 0)
                    noise_pred = model.forward_cfg(x_alpha_inp, tt_inp, cond)#.sample
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
            x_alpha = x_alpha + (alpha_start - alpha_end).view(-1, 1, 1, 1) * noise_pred

        return x_alpha

