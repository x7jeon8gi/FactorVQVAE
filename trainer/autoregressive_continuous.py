import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import RankLoss
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from module.continuous_transformer import ContinuousTransformer
from utils import get_root_dir, calc_ic

class continuousGPT(pl.LightningModule):
    def __init__(self, config, n_train_samples, ckpt_path=None):
        super().__init__()
        self.config = config
        self.input_channel = config['vqvae']['input_channel']
        self.num_features = config['vqvae']['num_features']
        self.hidden_size = config['vqvae']['hidden_size']
        self.num_factors = config['vqvae']['num_factors']
        self.dropout = config['vqvae']['dropout']
        # T_max: total number of steps for the cosine scheduler
        self.T_max = config['train']['num_epochs'] * np.ceil(n_train_samples / config['train']['batch_size'] + 1)
        
        # Instantiate the VAE+Transformer model.
        self.cont_mingpt = ContinuousTransformer(
            config=config,
            ckpt=ckpt_path
        )
        
        tf_hidden = config['transformer']['hidden_size']
        tf_head = config['transformer']['heads']
        tf_layers = config['transformer']['n_layers']
        seed = config['train']['seed']
        vq_hidden = config['vqvae']['hidden_size']
        vq_elements = config['vqvae']['num_elements']
        vq_code = config['vqvae']['num_factors']
        self.name = f'Ablation_conti2_VAE_{vq_code}_h{vq_hidden}_e{vq_elements}__Th_{tf_hidden}_h{tf_head}_l{tf_layers}_sd{seed}'
        
        self.ic = []   # for storing daily IC
        self.ric = []  # for storing daily RIC
        self.best_val_loss = float('inf')
        self.best_metrics_at_min_loss = {}
        self.eta = config['transformer']['eta']
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['train']['learning_rate'])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max)
        sch_config = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [sch_config]
    
    def forward(self, firm_char, inputs, market):
        """
        Forward pass.
          firm_char: firm-level input (예: (B, ..., feature_dim))
          inputs: Tensor of shape (B, seq_len, input_channel) — 재구성 대상 데이터
          market: (Optional) market 정보 (B, seq_len, market_dim)
        반환:
          y_hat: 재구성된 출력 (B, seq_len, num_elements)
          loss_dict: {'recon_loss': ..., 'kl_divergence': ..., 'loss': ...}
        """
        y_hat, loss_dict = self.cont_mingpt(firm_char=firm_char, y=inputs, market=market)
        return y_hat, loss_dict

    def training_step(self, batch, batch_idx):
        if batch.nelement() == 0:
            # 빈 배치는 스킵합니다.
            return None
        
        # 데이터 분할 (입력 구조에 맞게 수정)
        firm_char = batch[:, :, 0:158]              # firm-level 정보
        y = batch[:, :, 158].unsqueeze(-1)            # 재구성 대상, 연속 값
        market = batch[:, :, 159:]                    # market 정보
        
        y_hat, loss_dict = self.forward(firm_char, y, market)
        loss = loss_dict['loss']
        self.log('train_loss', loss)
        self.log('train_recon_loss', loss_dict['recon_loss'])
        #self.log('train_kl_div', loss_dict['kl_divergence'])
        return loss

    def validation_step(self, batch, batch_idx):
        firm_char = batch[:, :, 0:158]
        y = batch[:, :, 158].unsqueeze(-1)
        market = batch[:, :, 159:]
        
        y_hat, loss_dict = self.forward(firm_char, y, market)
        loss = loss_dict['loss']
        self.log('val_loss', loss, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_recon_loss', loss_dict['recon_loss'], on_epoch=True, logger=True, sync_dist=True)
        #self.log('val_kl_div', loss_dict['kl_divergence'], on_epoch=True, logger=True, sync_dist=True)
        
        # 마지막 시점의 예측값으로 IC, RIC 계산 (calc_ic 함수는 별도로 정의되어 있다고 가정)
        daily_ic, daily_ric = calc_ic(
            y_hat[:, -1].squeeze().detach().cpu().numpy(),
            y[:, -1].squeeze().detach().cpu().numpy()
        )
        self.ic.append(daily_ic)
        self.ric.append(daily_ric)
        
        return {"val_loss": loss, 
                "val_recon_loss": loss_dict['recon_loss'], }
                #"val_kl_div": loss_dict['kl_divergence']}
    
    def on_train_epoch_end(self):
        train_loss_epoch = self.trainer.callback_metrics.get('train_loss')
        if train_loss_epoch is not None:
            self.log('train_loss_epoch', train_loss_epoch, on_step=False, on_epoch=True, logger=True, sync_dist=True)
    
    def on_validation_epoch_end(self):
        # IC 및 RIC의 평균과 ICIR, RICIR 계산
        current_ic = np.mean(self.ic)
        current_ric = np.mean(self.ric)
        current_icir = current_ic / np.std(self.ic) if np.std(self.ic) != 0 else 0
        current_ricir = current_ric / np.std(self.ric) if np.std(self.ric) != 0 else 0

        metric = {
            'Val_IC': current_ic,
            'Val_ICIR': current_icir,
            'Val_RIC': current_ric,
            'Val_RICIR': current_ricir,
        }
        self.log_dict(metric, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.ic = []
        self.ric = []

        val_loss_epoch = self.trainer.callback_metrics.get('val_loss')
        if val_loss_epoch is not None and val_loss_epoch < self.best_val_loss:
            self.best_val_loss = val_loss_epoch
            self.best_metrics_at_min_loss = {
                'Best_Val_Loss': float(val_loss_epoch),
                'Best_Val_IC': current_ic,
                'Best_Val_ICIR': current_icir,
                'Best_Val_RIC': current_ric,
                'Best_Val_RICIR': current_ricir,
            }
            self.log_dict(self.best_metrics_at_min_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        if val_loss_epoch is not None:
            self.log('val_loss_epoch', val_loss_epoch, on_step=False, on_epoch=True, logger=True, sync_dist=True)