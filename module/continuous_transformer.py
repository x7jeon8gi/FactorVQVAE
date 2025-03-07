import torch
import torch.nn as nn
import torch.nn.functional as F
from module.variational_mingpt import LatentTransformer, GPTConfig
import numpy as np
from pathlib import Path
from module.vqvae import FactorDecoder, FeatureExtractor
from module.variational_ae import FactorEncoder
import os
import sys
from utils import freeze, get_root_dir, load_pretrained_tok_emb
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

class ContinuousTransformer(nn.Module):
    """
    Ablation study: Modified Transformer for a sequential VAE model.
    This model focuses on the MSE reconstruction loss instead of the next token prediction, 
    The encoder computes the mean (z_mu) and log variance (z_logvar) of the latent variable at each sequence timestep,
    Reparameterization to sample the continuous latent variable z.
    """
    def __init__(self,
                 config,
                 ckpt):
        super().__init__()

        self.config = config
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
            num_elements= config['vqvae']['num_elements']) # num_factors = num_elements
        
        # load trained models for encoder, decoder, and quantizer
        self.load_pretrained_model(ckpt)

        # Initialize transformer
        self.vocab_size = self.config['vqvae']['num_factors']
        self.pkeep = self.config['transformer']['pkeep']
        
        transformer_config = GPTConfig(
            vocab_size=0,  # 사용하지 않으므로 0 또는 임의의 값
            block_size=config['transformer']['num_tokens'] + 1,
            n_layer = 2,#config['transformer']['n_layers'],
            n_head = 2, #config['transformer']['heads'],
            n_embd = config['transformer']['hidden_size'],
            embd_pdrop=0.1,#config['transformer']['attn_pdrop'],
            resid_pdrop=0.1,#config['transformer']['attn_pdrop'],
            attn_pdrop=config['transformer']['attn_pdrop'],
            n_unmasked=config['transformer'].get('n_unmasked', 0)
        )
        transformer_config.market_dim = config['transformer']['hidden_size']
        
        # LatentTransformer: 입력은 continuous latent (B, T, dim)
        self.linear = nn.Linear(config['transformer']['hidden_size'], self.dim)
        self.latent_transformer = LatentTransformer(transformer_config, input_dim=self.dim)
        
        # Use Market and MarketAttention
        self.use_market = config['transformer']['use_market']
        self.market_extractor = FeatureExtractor(num_latent = config['vqvae']['market_features'],
                                                 hidden_size = config['transformer']['hidden_size'])

    def load_pretrained_model(self, ckpt):
        saved_model = ckpt
        saved_model = f"{saved_model}.ckpt" if saved_model and not saved_model.endswith('.ckpt') else saved_model
        checkpoint_path = Path(get_root_dir()).joinpath('temp/ablation1', saved_model)
        checkpoint = torch.load(checkpoint_path)['state_dict']

        def load_state_dict(module, prefix):
            state_dict = {k.replace(f'{prefix}.', ''): v for k, v in checkpoint.items() if k.startswith(prefix)}
            module.load_state_dict(state_dict)

        load_state_dict(self.feature_extractor, 'feature_extractor')
        load_state_dict(self.encoder, 'encoder')
        load_state_dict(self.decoder, 'decoder')

        freeze(self.encoder)
        freeze(self.feature_extractor)

        self.encoder.eval()
        self.feature_extractor.eval()
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def prepare_inputs(self, z_mu, device):
        sos_tokens = torch.zeros(z_mu.size(0),1,z_mu.size(2), dtype=torch.float).to(device)
        z_mu = torch.cat([sos_tokens, z_mu], dim=1)
        return z_mu

    @torch.no_grad()
    def encode(self, y):
        """
        Encodes input `y` into quantized representation `z`.
        """
        z_mu, z_logvar = self.encoder(y)
        return z_mu, z_logvar
    
    def forward(self, firm_char, y, market):
        """
        Args:
            firm_char: Tensor, firm-level input (예: (B, ...) 형태)
            y: Tensor of shape (B, seq_len, input_channel), 재구성 대상 데이터.
            market: (Optional) 시장 정보 (B, seq_len, market_dim)
        Returns:
            y_hat: 재구성된 출력.
            loss_dict: MSE 재구성 손실 및 KL divergence를 포함한 손실 dictionary.
        """
        firm_features = self.feature_extractor(firm_char)
        market_features = self.market_extractor(market)
        
        # Encode `y` into discrete representations
        z_mu, z_logvar = self.encode(y)  # (B, seq_len, hidden_size)
        z = self.reparameterize(z_mu, z_logvar)

        input_indices = self.prepare_inputs(z_mu, device=y.device)

        #  Transformer: latent 값들을 입력으로 받아 sequence 관계를 모델링
        if self.use_market and market is not None:
            market_features = self.market_extractor(market)
            logits = self.latent_transformer(input_indices[:,:-1], market=market_features)
        else:
            logits = self.latent_transformer(input_indices[:,:-1])
        
        # Decoder: firm feature와 Transformer의 출력을 결합하여 재구성
        
        z_transformed = self.linear(logits)
        y_hat, _ = self.decoder(firm_char=firm_features, inputs=z_transformed)
        
        # 손실 계산: 재구성 손실 (MSE) + KL divergence
        recon_loss = F.mse_loss(y_hat, y)
        #kl_div = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        total_loss = recon_loss #+ kl_div
        loss_dict = {'recon_loss': recon_loss, 'loss': total_loss}
        
        return y_hat, loss_dict