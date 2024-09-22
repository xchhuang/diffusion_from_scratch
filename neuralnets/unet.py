import math
import torch
import torch.nn as nn
from neuralnets.layers import MultiheadAttention, get_timestep_embedding, Normalize, nonlinearity, Upsample, Downsample

num_frames = 16
use_temporal = True


# class MultiheadAttention(nn.Module):
#     """
#     Multi-head attention layer without masking, with safe softmax.
#     """
#     def __init__(self, embed_dim, num_heads, dropout=0.0):
#         super(MultiheadAttention, self).__init__()
#         assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, query, key, value):
#         """
#         Forward pass for multi-head attention.

#         Args:
#             query: Tensor of shape (batch_size, seq_len, embed_dim)
#             key:   Tensor of shape (batch_size, seq_len, embed_dim)
#             value: Tensor of shape (batch_size, seq_len, embed_dim)

#         Returns:
#             Tensor of shape (batch_size, seq_len, embed_dim)
#         """

#         batch_q, seq_q, embed_q = query.size()
#         batch_kv, seq_kv, embed_kv = key.size()

#         Q = query.view(batch_q, seq_q, self.num_heads, self.head_dim)
#         K = key.view(batch_kv, seq_kv, self.num_heads, self.head_dim)
#         V = value.view(batch_kv, seq_kv, self.num_heads, self.head_dim)

#         Q = Q.transpose(1, 2)
#         K = K.transpose(1, 2)
#         V = V.transpose(1, 2)

#         # Scaled dot-product attention
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
#         attn_scores = attn_scores / (self.head_dim ** 0.5)
#         # Safe softmax
#         attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True)[0]  # Subtract max for numerical stability
#         attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
#         attn_weights = self.dropout(attn_weights)
#         attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

#         attn_output = attn_output.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
#         attn_output = attn_output.contiguous().view(batch_q, seq_q, embed_q)  # (batch_size, seq_len, embed_dim)

#         return attn_output




# def get_timestep_embedding(timesteps, embedding_dim):
#     """
#     This matches the implementation in Denoising Diffusion Probabilistic Models:
#     From Fairseq.
#     Build sinusoidal embeddings.
#     This matches the implementation in tensor2tensor, but differs slightly
#     from the description in Section 3.5 of "Attention Is All You Need".
#     """
#     assert len(timesteps.shape) == 1

#     half_dim = embedding_dim // 2
#     emb = math.log(10000) / (half_dim - 1)
#     emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
#     emb = emb.to(device=timesteps.device)
#     emb = timesteps.float()[:, None] * emb[None, :]
#     emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
#     if embedding_dim % 2 == 1:  # zero pad
#         emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
#     return emb


# def nonlinearity(x):
#     # swish
#     # return x*torch.sigmoid(x)
#     return nn.functional.silu(x)


# def Normalize(in_channels):
#     return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, affine=True)


# class Upsample(nn.Module):
#     def __init__(self, in_channels, with_conv):
#         super().__init__()
#         self.with_conv = with_conv
#         if self.with_conv:
#             self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         x = torch.nn.functional.interpolate(
#             x, scale_factor=2.0, mode="nearest")
#         if self.with_conv:
#             x = self.conv(x)
#         return x


# class Downsample(nn.Module):
#     def __init__(self, in_channels, with_conv):
#         super().__init__()
#         self.with_conv = with_conv
#         if self.with_conv:
#             # no asymmetric padding in torch conv, must do it ourselves
#             self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

#     def forward(self, x):
#         if self.with_conv:
#             pad = (0, 1, 0, 1)
#             x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
#             x = self.conv(x)
#         else:
#             x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
#         return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # 1D temporal convolutions
        self.norm1d_1 = Normalize(out_channels)
        self.conv1d_1 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1d_2 = Normalize(out_channels)
        self.conv1d_2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x, temb):
        
        temb_proj = self.temb_proj(nonlinearity(temb))

        h = x   # b,c,f,h,w
        h = self.norm1(h)   # norm goes first
        h = nonlinearity(h)
        h = self.conv1(h)

        batch_frame, channel, height, width = h.shape
        batch = int(batch_frame // num_frames)
        # print('x:', x.shape)
        h = h.reshape(batch, num_frames, channel, height, width).permute(0, 2, 1, 3, 4) # b,c,f,h,w
        
        if use_temporal:
            h = h.permute(0, 3, 4, 1, 2).reshape(batch*height*width, channel, num_frames)   # bhw,c,f
            h = self.norm1d_1(h)
            h = nonlinearity(h)
            h = self.conv1d_1(h)
            h = h.reshape(batch, height, width, channel, num_frames)
            h = h.permute(0, 3, 4, 1, 2) # b,c,f,h,w
        # print('h:', h.shape, temb_proj.shape)
        h = h + temb_proj[:, :, None, None, None]

        h = h.permute(0, 2, 1, 3, 4).reshape(batch_frame, channel, height, width)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # print('h:', h.shape, x.shape)
        if use_temporal:
            batch_frame, channel, height, width = h.shape
            batch = int(batch_frame // num_frames)
            h = h.reshape(batch, num_frames, channel, height, width).permute(0, 3, 4, 2, 1)
            h = h.reshape(batch*height*width, channel, num_frames)
            h = self.norm1d_2(h)
            h = nonlinearity(h)
            h = self.conv1d_2(h)
            h = h.reshape(batch, height, width, channel, num_frames).permute(0, 4, 3, 1, 2)
            h = h.reshape(batch_frame, channel, height, width)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        # print('x:', x.shape, 'h:', h.shape)
        x = x + h

        return x


class AttnBlock(nn.Module):
    def __init__(self, in_channels, nhead=8):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        # self.q_cond = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # self.k_cond = torch.nn.Linear(1024, in_channels)
        # self.v_cond = torch.nn.Linear(1024, in_channels)

        dropout = 0.1
        self.self_attn = MultiheadAttention(in_channels, nhead, dropout=dropout)
        # self.cross_attn = MultiheadAttention(in_channels, nhead, dropout=dropout)
        
        self.self_norm = torch.nn.LayerNorm(in_channels)
        # self.cross_norm = torch.nn.LayerNorm(in_channels)

        # 1D temporal attention
        self.norm1d = Normalize(in_channels)
        self.q1d = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k1d = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v1d = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.proj_out1d = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # self.q1d_cond = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.self_attn1d = MultiheadAttention(in_channels, nhead, dropout=dropout)
        # self.cross_attn1d = MultiheadAttention(in_channels, nhead, dropout=dropout)
        self.self_norm1d = torch.nn.LayerNorm(in_channels)
        # self.cross_norm1d = torch.nn.LayerNorm(in_channels)
        

    def forward(self, x, cond):
        
        # print('inp:', x.shape)
        h_ = x
        h_ = self.norm(h_)

        # self attention
        q = self.q(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w).permute(0, 2, 1)   # b,hw,c
        k = self.k(h_).reshape(b, c, h*w).permute(0, 2, 1)  # b,hw,c
        v = self.v(h_).reshape(b, c, h*w).permute(0, 2, 1)  # b,hw,c
        # print('q:', q.shape, 'k:', k.shape, 'v:', v.shape)
        self_attn = self.self_attn(q, k, v)[0]  # b,hw,c
        # print('self_attn:', self_attn.shape, h_.shape)
        h_ = h_.reshape(b, c, h*w).permute(0, 2, 1)   # b,hw,c
        h_ = self.self_norm(h_ + self_attn)
        h_ = h_.permute(0, 2, 1).reshape(b, c, h, w)
        h_ = self.proj_out(h_)

        
        if use_temporal:
            batch_frame, channel, height, width = h_.shape
            batch = int(batch_frame // num_frames)
            h_ = h_.reshape(batch, num_frames, channel, height, width).permute(0, 3, 4, 1, 2) # b,h,w,c,f
            h_ = h_.reshape(batch*height*width, channel, num_frames) # bhw,c,f
            h_ = self.norm1d(h_)    # bf,c,h,w
            # print('h_:', h_.shape)
            q = self.q1d(h_).permute(0, 2, 1)   # bhw,f,c
            k = self.k1d(h_).permute(0, 2, 1)   # bhw,f,c
            v = self.v1d(h_).permute(0, 2, 1)   # bhw,f,c
            self_attn1d = self.self_attn1d(q, k, v)[0]
            # print('self_attn1d:', self_attn1d.shape, h_.shape)
            h_ = self.self_norm1d(h_.permute(0, 2, 1) + self_attn1d).permute(0, 2, 1)   # bhw,c,f

            h_ = self.proj_out1d(h_)

            h_ = h_.reshape(batch, height, width, channel, num_frames).permute(0, 4, 3, 1, 2)   # b,f,c,h,w
            h_ = h_.reshape(batch_frame, channel, height, width)
            # print('h_:', h_.shape)

        if False:
            # cross attention
            q = self.q_cond(h_).reshape(b, c, h*w).permute(0, 2, 1)   # b,hw,c
            k = self.k_cond(cond)   # b,l,c
            v = self.v_cond(cond)   # b,l,c
            # print('q:', q.shape, 'k:', k.shape, 'v:', v.shape, cond.shape)
            # k = k.repeat(num_frames, 1, 1)
            # v = v.repeat(num_frames, 1, 1)
            cross_attn = self.cross_attn(q, k.repeat(num_frames, 1, 1), v.repeat(num_frames, 1, 1))[0]
            # print('cross_attn:', cross_attn.shape, h_.shape)
            h_ = h_.reshape(b, c, h*w).permute(0, 2, 1)   # b,hw,c
            h_ = self.cross_norm(h_ + cross_attn)
            h_ = h_.permute(0, 2, 1).reshape(b, c, h, w)
            h_ = self.proj_out(h_)
            # print('h_:', h_.shape)

            batch_frame, channel, height, width = h_.shape
            batch = int(batch_frame // num_frames)
            h_ = h_.reshape(batch, num_frames, channel, height, width).permute(0, 3, 4, 2, 1) # b,h,w,c,f
            h_ = h_.reshape(batch*height*width, channel, num_frames) # bhw,c,f
            # print('h_:', h_.shape)
            q = self.q1d_cond(h_).permute(0, 2, 1)   # bhw,f,c
            # print('qkv:', q.shape, k.shape, v.shape)
            cross_attn1d = self.cross_attn1d(q, k.repeat(height*width, 1, 1), v.repeat(height*width, 1, 1))[0]
            # print('cross_attn1d:', cross_attn1d.shape, h_.shape)
            h_ = self.cross_norm1d(h_.permute(0, 2, 1) + cross_attn1d).permute(0, 2, 1)   # bhw,c,f
            h_ = self.proj_out1d(h_)
            h_ = h_.reshape(batch, height, width, channel, num_frames).permute(0, 4, 3, 1, 2)   # b,f,c,h,w
            # print('h_:', h_.shape)

        # o = self.proj_out(h_)
        o = h_
        
        return o


class Unet(nn.Module):
    def __init__(self, ch_mult, resolution):
        super().__init__()
        # self.config = config
        ch = 128
        out_ch = 4
        # ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        # ch_mult = (1, 2, 2, 2)
        num_res_blocks = 2  #config.model.num_res_blocks
        attn_resolutions = [16,]    #config.model.attn_resolutions
        dropout = 0.1   #config.model.dropout
        in_channels = 4 #config.model.in_channels
        resolution = resolution #config.data.image_size
        resamp_with_conv = True #config.model.resamp_with_conv
        # num_timesteps = config.diffusion.num_diffusion_timesteps
        
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(self.ch, self.temb_ch),
            nn.SiLU(),
            nn.Linear(self.temb_ch, self.temb_ch),
        )
        
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult

        # print('in_ch_mult:', in_ch_mult, ch_mult)

        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            # print('i_level:', i_level, block_in, block_out)
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:    # 16
                    # print('curr_res:', curr_res, attn_resolutions)
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:   # no downsampling on last level
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, cond=None):
        
        assert x.shape[-2] == x.shape[-1] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.time_mlp(temb)
        
        # print('x:', x.shape, 'temb:', temb.shape, 'cond:', cond.shape)
        b, f, c, h, w = x.shape
        x = x.reshape(b*f, c, h, w)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:

                    # print('h:', h.shape, cond.shape)

                    h = self.down[i_level].attn[i_block](h, cond)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]  # (b, c, 4, 4)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h, cond)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, cond)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        # print('h:', h.shape)
        batch_frame, channel, height, width = h.shape
        batch = int(batch_frame // num_frames)
        h = h.reshape(batch, num_frames, channel, height, width)

        return h
    

    def forward_cfg(self, x, t, cond):

        # print('x:', x.shape, 't:', t.shape, 'cond:', cond.shape)
        out = self.forward(x, t, cond)
        split_channels = int(out.shape[0] // 2)
        cond_out = out[:split_channels]
        uncond_out = out[split_channels:]
        # new_out = uncond_out + cfg_scale * (cond_out - uncond_out)
        return torch.cat([uncond_out, cond_out], 0)






def get_unet(unet_name, resolution=32):
    ch_mult = (1, 2, 2, 2)
    if unet_name == 'unet_small':
        ch_mult = (1, 2, 2, 2)  # 44542980
    elif unet_name == 'unet_medium':
        ch_mult = (1, 2, 2, 4)  # 86647044
    elif unet_name == 'unet_large':
        ch_mult = (1, 2, 2, 2, 4)   # 98528516
    else:
        raise ValueError(f"Unknown unet_name: {unet_name}")
    model = Unet(ch_mult=ch_mult, resolution=resolution)
    return model