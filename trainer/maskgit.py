import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from module.bidirectional_transformer import BidirectionalTransformer

class MaskGIT(pl.LightningModule):
    def __init__(self, 
                config, 
                n_train_samples, 
                ode_func=None,
                ckpt_path=None, 
                ignore_keys=list()):
        super().__init__()
        self.config = config
        self.input_channel = config['vqvae']['input_channel']
        self.num_features = config['vqvae']['num_features']
        self.hidden_size = config['vqvae']['hidden_size']
        self.num_factors = config['vqvae']['num_factors']
        self.dropout = config['vqvae']['dropout']
        
        self.maskgit = BidirectionalTransformer(temperature= config['transformer']['temperature'],
                                                T = config['transformer']['T'],
                                                config= config)
        
        self.T_max = config['train']['num_epochs'] * (np.ceil(n_train_samples / config['train']['batch_size']) + 1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr= self.config['train']['learning_rate'])
        scheduler = CosineAnnealingLR(optimizer, T_max= self.T_max)
        sch_config = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [sch_config]

    def forward(self, firm_char, inputs):
        logit, target = self.maskgit(firm_char, inputs)
        return logit, target
    
    @torch.no_grad()
    def _plot_reconstructions(self, input, firm_char):
        r = np.random.rand()
        
        if self.training and r <= 0.05:
            self.maskgit.eval()
            s = self.maskgit.iterative_decoding()
            yhat = self.maskgit.decode_token_idx_to_timeseries(firm_char= firm_char, s = s)
            b = np.random.randint(0, yhat.shape[0])
            
            fig, axes = plt.subplots(1, 1, figsize=(6, 3))
            plt.suptitle(f'epoch_{self.current_epoch}')
            axes.plot(input[b].cpu().squeeze())
            axes.plot(yhat[b].detach().cpu().squeeze())
            axes.set_title('y vs yhat (stage II)')
            axes.set_ylim(-3, 3)

            plt.tight_layout()
            wandb.log({"y vs yhat (stage II)": wandb.Image(plt)})
            plt.close()
            self.maskgit.train()

    def training_step(self, batch, batch_idx):
        firm_char = batch[:, : ,0:158]
        y = batch[:, :, 158].unsqueeze(-1)
        market = batch[:, :, 159:]

        logit, target = self.forward(firm_char, y)
        prior_loss = F.cross_entropy(logit.reshape(-1, logit.size(-1)), target.reshape(-1))
        self.log('train_loss', prior_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        # maskgit sampling
        # self._plot_reconstructions(inputs, firm_char)
        return prior_loss

    def validation_step(self, batch, batch_idx):
        firm_char = batch[:, : ,0:158]
        y = batch[:, :, 158].unsqueeze(-1)
        market = batch[:, :, 159:]
        logit, target = self.forward(firm_char, y)
        prior_loss = F.cross_entropy(logit.reshape(-1, logit.size(-1)), target.reshape(-1))
        self.log('val_loss', prior_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return prior_loss
    
    def on_train_epoch_end(self):
        train_loss_epoch = self.trainer.callback_metrics.get('train_loss')
        if train_loss_epoch is not None:
            self.log('train_loss_epoch', train_loss_epoch, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def on_validation_epoch_end(self):
        val_loss_epoch = self.trainer.callback_metrics.get('val_loss')
        if val_loss_epoch is not None:
            self.log('val_loss_epoch', val_loss_epoch, on_step=False, on_epoch=True, logger=True, sync_dist=True)