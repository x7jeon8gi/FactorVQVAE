import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from vqtorch.nn import VectorQuant
from module.vqvae import FactorDecoder, FeatureExtractor
from module.variational_ae import FactorEncoder
import pytorch_lightning as pl
import matplotlib.pyplot as plt

def compute_loss(reconstruction, y, z_mu, z_logvar, current_epoch, total_epochs, recon_weight=1.0):
    # 재구성 손실: MSE
    recon_loss = F.mse_loss(reconstruction, y)
    
    # KL divergence 계산 (배치별 평균)
    kl_div = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp()) / y.size(0)
    
    # KL annealing: current_epoch가 total_epochs에 도달할 때까지 β 값을 선형 증가시킴 (최대 1)
    beta = min(1.0, current_epoch / total_epochs)
    
    total_loss = recon_weight * recon_loss + beta * kl_div
    return total_loss, recon_loss, kl_div


class FactorSequentialVAE(pl.LightningModule):
    """
    FactorSequentialVAE is a class designed for ablation studies, specifically 
    for experiments that do not use Vector Quantization (VQ) in the Variational 
    Autoencoder (VAE). This version has been modified to incorporate the 
    KL divergence term, thus functioning as a standard VAE.
    
    Attributes:
        config (dict): Configuration dictionary containing model parameters.
        n_train_samples (int): Number of training samples.
        ckpt_path (str, optional): Path to the checkpoint file.
        ignore_keys (list, optional): List of keys to ignore during initialization.
        input_channel (int): Number of input channels.
        num_features (int): Number of features (alpha 158).
        hidden_size (int): Size of the hidden layer (128).
        num_elements (int): Number of elements (64).
        dropout (float): Dropout rate (0.1).
        num_heads (int): Number of attention heads.
        alpha (float): Alpha parameter for the model.
        T_max (int): Maximum number of epochs for training.
    
    Methods:
        __init__: Initializes the FactorSequentialVAE with the given configuration.
        init_from_ckpt: Initializes model weights from a checkpoint.
        decode: Performs the decoding operation.
        forward: Computes the forward pass with reconstruction loss and KL divergence.

    """
    def __init__(self,
                 config,
                 n_train_samples,
                 ckpt_path=None,
                 ignore_keys=list()):
        super().__init__()
        self.config = config
        self.input_channel = config['vqvae']['input_channel']   # 예: 1
        self.num_features  = config['vqvae']['num_features']      # 예: 158
        self.hidden_size   = config['vqvae']['hidden_size']       # 예: 128
        self.num_elements  = config['vqvae']['num_elements']       # 예: 64
        self.dropout       = config['vqvae']['dropout']            # 예: 0.1
        self.num_heads     = config['vqvae']['num_heads']
        self.alpha         = config['vqvae']['alpha']
        self.T_max         = config['train']['num_epochs'] * (np.ceil(n_train_samples / config['train']['batch_size']) + 1)

        # Feature extractor remains unchanged.
        self.feature_extractor = FeatureExtractor(num_latent=self.num_features,
                                                  hidden_size=self.hidden_size)

        # Modified encoder: it should output (mu, logvar)
        self.encoder = FactorEncoder(input_size=self.input_channel,
                                     hidden_size=self.hidden_size, 
                                     num_heads=self.num_heads,
                                     use_attn=True, 
                                     dropout=self.dropout)
        
        self.decoder = FactorDecoder(input_size=self.hidden_size,
                                     hidden_size=self.hidden_size,
                                     num_elements=self.num_elements)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def decode(self, firm_char, z):
        # Decoder returns the reconstruction and optionally other info (e.g., attention weights)
        reconstruction, _ = self.decoder(firm_char, z)
        return reconstruction
    
    def forward(self, firm_char, y):
        """
        Forward pass for the VAE.
        
        Args:
            firm_char: Input features for the feature extractor.
            y: Input data for the encoder and reconstruction target.
        
        Returns:
            reconstruction: The reconstructed output.
            loss_dict: Dictionary containing reconstruction loss, KL divergence, and total loss.
        """
        # Feature extraction remains the same.
        firm_char = self.feature_extractor(firm_char)
        
        # --- VAE Encoding ---
        # Modify the encoder to output mu and logvar
        # 예: z_mu, z_logvar = self.encoder(y)
        # 만약 기존 인코더 구현이 단일 출력을 반환한다면, 해당 부분을 변경해야 합니다.
        z_mu, z_logvar = self.encoder(y)
        
        # Reparameterization trick: sample z from N(z_mu, sigma^2)
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mu + eps * std
        
        # --- Decoding ---
        reconstruction = self.decode(firm_char, z)
        
        # --- Loss Calculation ---
        # Reconstruction loss (예: MSE)
        recon_loss = F.mse_loss(reconstruction, y)
        
        # KL Divergence loss: D_KL(q(z|y) || p(z)), assuming p(z) ~ N(0,I)
        # 계산 시, 배치 단위 평균 혹은 합산 여부는 문제에 따라 조정할 수 있습니다.
        kl_div = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp()) / y.size(0)
        
        total_loss = recon_loss + kl_div
        
        loss_dict = {'recon_loss': recon_loss, 'kl_divergence': kl_div, 'loss': total_loss}
        return reconstruction, loss_dict
    
    def configure_optimizers(self):
        optimizer  = torch.optim.AdamW(self.parameters(), lr=self.config['train']['learning_rate'])
        scheduler  = CosineAnnealingLR(optimizer, T_max=self.T_max)
        sch_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [sch_config]
    
    def training_step(self, batch, batch_idx):
        
        firm_char = batch[:, :, 0:158]
        y = batch[:, :, 158].unsqueeze(-1)
        
        reconstruction, loss_dict = self.forward(firm_char, y)
        
        self.log('train_loss', loss_dict['loss'], prog_bar=True)
        self.log('train_recon_loss', loss_dict['recon_loss'], prog_bar=True)
        self.log('train_kl_div', loss_dict['kl_divergence'], prog_bar=True)
        
        return loss_dict['loss']
    
    def validation_step(self, batch, batch_idx):
        
        firm_char = batch[:, :, 0:158]
        y = batch[:, :, 158].unsqueeze(-1)
        
        reconstruction, loss_dict = self.forward(firm_char, y)
        
        self.log('val_loss', loss_dict['loss'], prog_bar=True)
        self.log('val_recon_loss', loss_dict['recon_loss'], prog_bar=True)
        self.log('val_kl_div', loss_dict['kl_divergence'], prog_bar=True)
        
        return loss_dict['loss']
    
    def on_train_epoch_end(self):
        return super().on_train_epoch_end()
    
    def on_validation_epoch_end(self):
        return super().on_validation_epoch_end()