import torch
import torch.nn as nn
import math




def nonlinearity(x):
    # swish
    # return x*torch.sigmoid(x)
    return nn.functional.silu(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, affine=True)




class MultiheadAttention(nn.Module):
    """
    Multi-head attention layer without masking, with safe softmax.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Forward pass for multi-head attention.

        Args:
            query: Tensor of shape (batch_size, seq_len, embed_dim)
            key:   Tensor of shape (batch_size, seq_len, embed_dim)
            value: Tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim)
        """

        batch_q, seq_q, embed_q = query.size()
        batch_kv, seq_kv, embed_kv = key.size()

        Q = query.view(batch_q, seq_q, self.num_heads, self.head_dim)
        K = key.view(batch_kv, seq_kv, self.num_heads, self.head_dim)
        V = value.view(batch_kv, seq_kv, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        # Safe softmax
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True)[0]  # Subtract max for numerical stability
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

        attn_output = attn_output.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.contiguous().view(batch_q, seq_q, embed_q)  # (batch_size, seq_len, embed_dim)

        return attn_output




def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb





class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    
    

