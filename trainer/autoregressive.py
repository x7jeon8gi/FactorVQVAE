import os
import random
#from pytorch_lightning.utilities.types import OptimizerLRScheduler
# from torch.optim import Optimizer
# from torch.optim.lr_scheduler import _LRScheduler
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import RankLoss
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from module.gpt_transformer import AutoRegressiveTransformer
from utils import get_root_dir, calc_ic

class minGPT(pl.LightningModule):
    def __init__(
            self,
            config,
            n_train_samples,
        ):
        super().__init__()
        self.config = config
        self.input_channel = config['vqvae']['input_channel']
        self.num_features = config['vqvae']['num_features']
        self.hidden_size = config['vqvae']['hidden_size']
        self.num_factors = config['vqvae']['num_factors']
        self.dropout = config['vqvae']['dropout']
        self.sos_token_ids = config['vqvae']['num_factors'] # same as codebook size
        self.mingpt = AutoRegressiveTransformer(temperature= config['transformer']['temperature'],
                                                config= config)
        self.T_max = config['train']['num_epochs'] * np.ceil(n_train_samples / config['train']['batch_size']+1)
        self.rank_loss = config['transformer']['rank_loss']
        if self.rank_loss:
            alpha = config['transformer']['rank_loss_alpha']
            self.mse_loss = RankLoss(alpha=alpha)
        else:
            self.mse_loss = torch.nn.MSELoss()

        tf_hidden = config['transformer']['hidden_size']
        tf_head = config['transformer']['heads']
        tf_layers = config['transformer']['n_layers']
        seed = config['train']['seed']
        vq_hidden = config['vqvae']['hidden_size']
        vq_elements = config['vqvae']['num_elements']
        vq_code = config['vqvae']['num_factors']
        alpha = config['vqvae']['alpha']
        rank_alpha = config['transformer']['rank_loss_alpha']

        self.name = f'Revise2_{rank_alpha}_VQ_{vq_code}_h{vq_hidden}_e{vq_elements}__Th_{tf_hidden}_h{tf_head}_l{tf_layers}_sd{seed}' # !Auto
        self.ic = []
        self.ric = []
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr= self.config['train']['learning_rate'])
        scheduler = CosineAnnealingLR(optimizer, T_max= self.T_max)
        sch_config = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [sch_config]
    
    def forward(self, firm_char, inputs, market):
        logit, target, y_hat = self.mingpt(firm_char=firm_char, y=inputs, market= market)
        return logit, target, y_hat


    def training_step(self, batch, batch_idx):
        if batch.nelement() == 0:
            # Skip if the batch is empty
            return None
        
        firm_char = batch[:, : ,0:158]
        y = batch[:, :, 158].unsqueeze(-1)
        market = batch[:, :, 159:]
        logit, target, y_hat = self.forward(firm_char, y, market)
        prior_loss = F.cross_entropy(logit.reshape(-1, logit.size(-1)), target.reshape(-1))
        mse_loss = self.mse_loss(y_hat, y)
        loss = prior_loss + mse_loss
        self.log('train_loss', loss)
        self.log('train_prior_loss', prior_loss)
        self.log('train_mse_loss', mse_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        firm_char = batch[:, : ,0:158]
        y = batch[:, :, 158].unsqueeze(-1)
        market = batch[:, :, 159:]

        logit, target, y_hat = self.forward(firm_char, y, market)
        prior_loss = F.cross_entropy(logit.reshape(-1, logit.size(-1)), target.reshape(-1), ignore_index=-1)
        mse_loss = self.mse_loss(y_hat, y)
        loss = prior_loss + mse_loss
        self.log('val_loss', loss, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_prior_loss', prior_loss, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_mse_loss', mse_loss, on_epoch=True, logger=True, sync_dist=True)

        # 마지막 값(예측 값)에 대한 IC, RIC 계산
        daily_ic, daily_ric = calc_ic(y_hat[:,-1].squeeze().detach().cpu().numpy(), y[:,-1].squeeze().detach().cpu().numpy())
        self.ic.append(daily_ic)
        self.ric.append(daily_ric)
        return {"val_loss": loss, "val_prior_loss":prior_loss, "val_mse_loss":mse_loss}
    
    def on_train_epoch_end(self):
        train_loss_epoch = self.trainer.callback_metrics.get('train_loss')
        if train_loss_epoch is not None:
            self.log('train_loss_epoch', train_loss_epoch, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def on_validation_epoch_end(self):
        val_loss_epoch = self.trainer.callback_metrics.get('val_loss')
        metric = {
            'Val_IC': np.mean(self.ic),
            'Val_ICIR': np.mean(self.ric) / np.std(self.ric),
            'Val_RIC': np.mean(self.ric),
            'Val_RICIR': np.mean(self.ric) / np.std(self.ric)
        }
        self.log_dict(metric, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        # Reset the IC and RIC lists
        self.ic = []
        self.ric = []

        if val_loss_epoch is not None:
            self.log('val_loss_epoch', val_loss_epoch, on_step=False, on_epoch=True, logger=True, sync_dist=True)
