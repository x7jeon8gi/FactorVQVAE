import torch
import torch.nn as nn
import torch.nn.functional as F
from module.mingpt import GPT
import copy
import numpy as np
from pathlib import Path
from module.vqvae import FactorEncoder, FactorDecoder, FeatureExtractor
from vqtorch.nn import VectorQuant
import os
import sys
import math 
from utils import freeze, get_root_dir, load_pretrained_tok_emb

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

class AutoRegressiveTransformer(nn.Module):
    """
    #! For the most part, you should follow the minGPT implementation.
    #! However, be careful to slightly modify the tok_emb (token embedding).
    """
    def __init__(self,
                 temperature,
                 config):
        super().__init__()

        self.sos_token_ids = config['vqvae']['num_factors'] # sos token 

        self.config = config
        self.num_factors = config['vqvae']['num_factors']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # define models
        self.dim           = config['vqvae']['hidden_size']
        self.input_channel = config['vqvae']['input_channel']
        self.dropout       = config['vqvae']['dropout'] # 0.1
        self.num_heads     = config['vqvae']['num_heads']
        self.num_features  = config['vqvae']['num_features']

        self.feature_extractor = FeatureExtractor(
            num_latent  = self.num_features,
            hidden_size = self.dim)

        self.encoder = FactorEncoder(
            input_size  = self.input_channel, 
            hidden_size = self.dim, 
            num_heads   = self.num_heads,
            use_attn    = True,
            dropout     = self.dropout)

        self.decoder = FactorDecoder(
            input_size  = self.dim, hidden_size = self.dim,
            num_factors = config['vqvae']['num_elements']) # num_factors = num_elements
        
        self.quantizer = VectorQuant(
            feature_size = self.dim,                                 # feature dimension corresponding to the vectors
            num_codes    = self.num_factors,                         # number of codebook vectors
            beta         = self.config['quantizer']['beta'],         # (default: 0.9) commitment trade-off
            kmeans_init  = self.config['quantizer']['kmeans_init'],  # (default: False) whether to use kmeans++ init
            norm         = None,                                     # (default: None) normalization for the input vectors
            cb_norm      = None,                                     # (default: None) normalization for codebookc vectors
            affine_lr    = self.config['quantizer']['affine_lr'],    # (default: 0.0) lr scale for affine parameters
            sync_nu      = self.config['quantizer']['sync_nu'],      # (default: 0.0) codebook synchronization contribution
            replace_freq = self.config['quantizer']['replace_freq'], # (default: None) frequency to replace dead codes
            dim=-1,                                                  # (default: -1) dimension to be quantized
            )
        
        # load trained models for encoder, decoder, and quantizer
        self.load_pretrained_model(config)

        # Initialize transformer
        self.vocab_size = self.config['vqvae']['num_factors']
        self.pkeep = self.config['transformer']['pkeep']
        
        self.transformer = GPT(
            vocab_size = self.vocab_size,
            block_size = self.config['transformer']['num_tokens'] + 1,
            n_layer    = self.config['transformer']['n_layers'],
            n_head     = self.config['transformer']['heads'],
            n_embd     = self.config['transformer']['hidden_size'],
            market_dim = self.config['transformer']['hidden_size'], 
            attn_pdrop = self.config['transformer']['attn_pdrop'],
        )

        # Use Market and MarketAttention
        self.use_market = config['transformer']['use_market']
        self.market_extractor = FeatureExtractor(num_latent = config['vqvae']['market_features'],
                                                 hidden_size = config['transformer']['hidden_size'])

    def load_pretrained_model(self, config):
        saved_model = config['transformer']['saved_model']
        saved_model = f"{saved_model}.ckpt" if saved_model and not saved_model.endswith('.ckpt') else saved_model
        checkpoint_path = Path(get_root_dir()).joinpath('checkpoints', saved_model)
        checkpoint = torch.load(checkpoint_path)['state_dict']

        def load_state_dict(module, prefix):
            state_dict = {k.replace(f'{prefix}.', ''): v for k, v in checkpoint.items() if k.startswith(prefix)}
            module.load_state_dict(state_dict)

        load_state_dict(self.feature_extractor, 'feature_extractor')
        load_state_dict(self.encoder, 'encoder')
        load_state_dict(self.decoder, 'decoder')
        load_state_dict(self.quantizer, 'quantizer')

        freeze(self.encoder)
        freeze(self.quantizer)
        freeze(self.feature_extractor)

        self.encoder.eval()
        self.quantizer.eval()
        self.feature_extractor.eval()

    @torch.no_grad()
    def encode_to_z_q(self, y):
        """
        Encodes input `y` into quantized representation `z_q`.
        """
        z_e = self.encoder(y)
        z_q, vq_dict = self.quantizer(z_e)
        return z_q, vq_dict['q'].squeeze()

    @torch.no_grad()
    def prepare_transformer_inputs(self, target_indices, device):
        """
        Prepares input indices for the transformer, including SOS tokens and masked indices.
        """
        sos_tokens = torch.full((target_indices.shape[0], 1), self.sos_token_ids, dtype=torch.long, device=device)
        
        # Apply masking for denoising training (optional)
        assert 0.0 <= self.pkeep <= 1.0, "pkeep must be in the range [0, 1]"
        if self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep * torch.ones(target_indices.shape, device=device)).round().to(torch.int64)
            random_indices = torch.randint_like(target_indices, self.vocab_size, device=device)
            masked_indices = mask * target_indices + (1 - mask) * random_indices
        else:
            masked_indices = target_indices
        # Concatenate SOS tokens with masked indices
        input_indices = torch.cat((sos_tokens, masked_indices), dim=1)
        return input_indices
    
    def decode_quantized_embeddings(self, firm_features, predicted_indices):
        """
        Passes quantized embeddings through the decoder.
        """
        # Retrieve quantized embeddings from the codebook
        codebook = self.quantizer.get_codebook().to(self.device)
        quantized_embeddings = F.embedding(predicted_indices, codebook)

        # Decode the quantized embeddings
        y_hat, _ = self.decoder(firm_char=firm_features, inputs=quantized_embeddings)
        return y_hat
    
    def forward(self, firm_char, y, market):
        """
        Forward pass through the model.
        """
        device = firm_char.device
        
        # Extract features
        firm_features = self.feature_extractor(firm_char)
        market_features = self.market_extractor(market)
        
        # Encode `y` into discrete representations
        z_q, target_indices = self.encode_to_z_q(y)
        
        # Concatenate SOS tokens with masked indices
        input_indices = self.prepare_transformer_inputs(target_indices, device)
        
        # Generate transformer logits
        if self.use_market:
            logits = self.transformer(input_indices[:, :-1], market_features)
        else:
            logits = self.transformer(input_indices[:, :-1])
        
        # Compute predicted indices from logits
        predicted_indices = torch.argmax(logits, dim=-1)
        
        # Decode quantized embeddings
        y_hat = self.decode_quantized_embeddings(firm_features = firm_features, 
                                                 predicted_indices = predicted_indices)
        
        return logits, target_indices, y_hat
        
    # def top_k_logits(self, logits, k):
    #     v, ix = torch.topk(logits, k)
    #     out = logits.clone()
    #     out[out < v[..., [-1]]] = -float("inf")
    #     return out
    
    # @torch.no_grad()
    # def sample(self, x, c, steps, temperature=1.0, top_k=3):
    #     self.transformer.eval()
    #     x = torch.cat((c, x), dim=1)
    #     for k in range(steps):
    #         logits = self.transformer(x)
    #         logits = logits[:, -1, :] / temperature

    #         if top_k is not None:
    #             logits = self.top_k_logits(logits, top_k)

    #         probs = F.softmax(logits, dim=-1)

    #         ix = torch.multinomial(probs, num_samples=1)

    #         x = torch.cat((x, ix), dim=1)

    #     x = x[:, c.shape[1]:]
    #     self.transformer.train()
    #     return x