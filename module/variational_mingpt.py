import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# =================== GPT의 핵심 모듈 재사용 ===================

class GPTConfig:
    """ Base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size  # 본 모델에서는 사용되지 않음
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class CausalSelfAttention(nn.Module):
    """
    Vanilla multi-head masked self-attention layer with a projection at the end.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask: (1, 1, block_size, block_size)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, n_head, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y, (k, v)

class MarketAttention(nn.Module):
    """
    Market Attention layer for incorporating market information.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.market_dim, config.n_embd)
        self.value = nn.Linear(config.market_dim, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x, market):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(market).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(market).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.market_attn = MarketAttention(config)
        
    def forward(self, x, market=None, layer_past=None, return_present=False):
        attn_in = self.ln1(x)
        attn_out, present = self.attn(attn_in, layer_past=layer_past)
        x = x + attn_out

        if market is not None:
            market_in = self.ln2(x)
            market_out = self.market_attn(market_in, market)
            x = x + market_out

        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x

# =================== 연속적 latent 입력을 위한 Transformer ===================

class LatentTransformer(nn.Module):
    """
    Transformer for continuous latent representations.
    
    Inputs:
        x: Tensor of shape (B, T, input_dim) — VAE에서 추출된 연속 latent 값들.
    Outputs:
        Tensor of shape (B, T, n_embd), 변환된 latent 표현.
    """
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        # 입력 차원이 Transformer의 임베딩 차원과 다를 경우 투영
        if input_dim != config.n_embd:
            self.input_proj = nn.Linear(input_dim, config.n_embd)
        else:
            self.input_proj = nn.Identity()
        # Positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, x, market=None):
        B, T, _ = x.size()
        assert T <= self.block_size, "입력 시퀀스 길이가 block_size를 초과합니다."
        x = self.input_proj(x)  # (B, T, n_embd)
        pos_emb = self.pos_emb[:, :T, :]
        x = x + pos_emb
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, market=market)
        x = self.ln_f(x)
        return x