import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from vqtorch.nn import VectorQuant
from module.vqvae import FactorDecoder, FactorEncoder, FeatureExtractor
import pytorch_lightning as pl
import matplotlib.pyplot as plt
# from vector_quantize_pytorch import VectorQuantize

class FactorVQVAE(pl.LightningModule):
    def __init__(self,
                 config,
                 n_train_samples,
                 ckpt_path=None,
                 ignore_keys=list(),):
        
        super().__init__()
        self.config = config
        self.input_channel = config['vqvae']['input_channel'] # 1
        self.num_features  = config['vqvae']['num_features'] # alpha 168
        self.hidden_size   = config['vqvae']['hidden_size'] # 128
        self.num_factors   = config['vqvae']['num_factors'] # 32
        self.num_elements  = config['vqvae']['num_elements'] # 1
        self.dropout       = config['vqvae']['dropout'] # 0.1
        self.num_heads     = config['vqvae']['num_heads']
        self.alpha         = config['vqvae']['alpha']
        self.T_max         = config['train']['num_epochs'] * (np.ceil(n_train_samples / config['train']['batch_size']) + 1)

        self.feature_extractor = FeatureExtractor(num_latent = self.num_features,
                                                  hidden_size = self.hidden_size)

        self.encoder = FactorEncoder(input_size = self.input_channel,
                                     hidden_size= self.hidden_size, 
                                     num_heads= self.num_heads,
                                     use_attn= True, 
                                     dropout= self.dropout, 
                                     )
        
        self.decoder = FactorDecoder(input_size = self.hidden_size,
                                     hidden_size = self.hidden_size,
                                     num_factors= self.num_elements,)

        self.quantizer = VectorQuant(
                    feature_size=self.hidden_size,                          # feature dimension corresponding to the vectors
                    num_codes=self.num_factors,                             # number of codebook vectors
                    beta=self.config['quantizer']['beta'],                  # (default: 0.9) commitment trade-off
                    kmeans_init=self.config['quantizer']['kmeans_init'],    # (default: False) whether to use kmeans++ init
                    norm=None,                                              # (default: None) normalization for the input vectors
                    cb_norm=None,                                           # (default: None) normalization for codebookc vectors
                    affine_lr= self.config['quantizer']['affine_lr'],       # (default: 0.0) lr scale for affine parameters
                    sync_nu=self.config['quantizer']['sync_nu'],            # (default: 0.0) codebook synchronization contribution
                    replace_freq= self.config['quantizer']['replace_freq'], # (default: None) frequency to replace dead codes
                    dim= -1,                                                  # (default: -1) dimension to be quantized
                    )
        
        # self.quantizer = VectorQuantize(
        #     dim = self.hidden_size,
        #     codebook_size= self.num_factors,
        #     decay = 0.9,
        #     commitment_weight= 0.25,
        #     kmeans_init = True,
        #     kmeans_iters = self.r_freq,
        # ).cuda()

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.epoch_vq_codes = []

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

    def encode(self, y):
        z_e  = self.encoder(y)
        z_q, vq_dict = self.quantizer(z_e) # (batch_size, seq_len, hidden_size)

        return z_q, vq_dict['loss'], vq_dict #?(z, z_q, d, q, loss, perplexity)

        # z_q, indices, cmt_loss = self.quantizer(z_e)
        # return z_q, cmt_loss, indices
    
    def decode(self, firm_char, z_q):
        reconstruction, _  = self.decoder(firm_char, z_q)
        return reconstruction
    
    def forward(self, firm_char, y):
        recon_loss = {'recon_loss': 0}
        cmt_loss = {'cmt_loss': 0}

        # encode
        firm_char = self.feature_extractor(firm_char)
        
        z_q, cmt_loss['cmt_loss'], vq_dict = self.encode(y)
        # z_q, commit_loss, vq_dict = self.encode(y)
        # cmt_loss['cmt_loss'] = commit_loss

        # decode
        reconstruction = self.decode(firm_char, z_q)
        # calculate loss
        recon_loss['recon_loss'] = F.mse_loss(reconstruction, y)
        
        # plot `x` and `xhat`
        self._plot_reconstructions(y, reconstruction)

        return reconstruction, cmt_loss, recon_loss, vq_dict

    def _plot_reconstructions(self, y, yhat):
        r = np.random.rand()
        if self.training and r <= 0.05:
            b = np.random.randint(0, yhat.shape[0])
            
            fig, axes = plt.subplots(1, 1, figsize=(8, 5))
            plt.suptitle(f'epoch_{self.current_epoch}')
            axes.plot(y[b].cpu().squeeze())
            axes.plot(yhat[b].detach().cpu().squeeze())
            axes.set_title('y vs yhat')
            axes.legend(['y', 'yhat'])

            plt.tight_layout()
            wandb.log({"y vs yhat (training)": wandb.Image(plt)})
            plt.close()

    def calculate_loss(self, cmt_loss, recon_loss):
        loss = self.alpha * recon_loss['recon_loss'] + cmt_loss['cmt_loss']
        return loss
    
    def configure_optimizers(self):
        optimizer  = torch.optim.AdamW(self.parameters(), lr=self.config['train']['learning_rate'])
        scheduler  = CosineAnnealingLR(optimizer, T_max=self.T_max)
        sch_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [sch_config]

    def training_step(self, batch, batch_idx):

        firm_char = batch[:, :, 0:158]
        y = batch[:, :, 158].unsqueeze(-1)
        recon, cmt_loss, recon_loss, vq_dict = self.forward(firm_char, y)
        loss = self.calculate_loss(cmt_loss, recon_loss)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_recon_loss', recon_loss['recon_loss'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_cmt_loss', cmt_loss['cmt_loss'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        # Collect vq_dict['q'] values
        if isinstance(vq_dict['q'], torch.Tensor):
            self.epoch_vq_codes.append(vq_dict['q'].detach().cpu().numpy())
    
        # if isinstance(vq_dict, torch.Tensor):
        #     self.epoch_vq_codes.append(vq_dict.detach().cpu().numpy())

        return {"loss": loss, "recon_loss": recon_loss, "cmt_loss": cmt_loss}
    
    def validation_step(self, batch, batch_idx):
        # data = torch.squeeze(batch, dim=0)
        
        firm_char = batch[:, :, 0:158]
        y = batch[:, :, 158].unsqueeze(-1)
        recon, cmt_loss, recon_loss, vq_dict = self.forward(firm_char, y)
        loss = self.calculate_loss(cmt_loss, recon_loss)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_recon_loss', recon_loss['recon_loss'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_cmt_loss', cmt_loss['cmt_loss'], on_step=True, on_epoch=True, logger=True, sync_dist=True)

        return {"loss": loss, "recon_loss": recon_loss, "cmt_loss": cmt_loss}
    
    def on_train_epoch_end(self):
        # print(f"Epoch {self.current_epoch} ended.")  # Debugging log
        
        train_loss_epoch = self.trainer.callback_metrics.get('train_loss')
        if train_loss_epoch is not None:
            self.log('train_loss_epoch', train_loss_epoch, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        if self.epoch_vq_codes:
            # Concatenate all collected vq_codes for the epoch
            epoch_vq_codes_np = np.concatenate(self.epoch_vq_codes, axis=0)
            
            # Calculate unique value counts
            unique, counts = np.unique(epoch_vq_codes_np, return_counts=True)
            unique_counts = {int(k): int(v) for k, v in zip(unique, counts)}
            
            # Plot bar graph
            plt.figure(figsize=(25, 5))
            plt.bar(unique_counts.keys(), unique_counts.values())
            plt.xlabel('Codebook Index')
            plt.ylabel('Frequency')
            plt.title(f'Codebook Utilization at Epoch {self.current_epoch}')
            plt.xticks(list(unique_counts.keys()))
            plt.grid(True)
            
            # Log bar graph using wandb
            wandb.log({"Codebook Utilization": wandb.Image(plt)})
            
            #* Clear the list for the next epoch
            self.epoch_vq_codes = []
            plt.close()

    def on_validation_epoch_end(self):
        val_loss_epoch = self.trainer.callback_metrics.get('val_loss')
        if val_loss_epoch is not None:
            self.log('val_loss_epoch', val_loss_epoch, on_step=False, on_epoch=True, logger=True, sync_dist=True)