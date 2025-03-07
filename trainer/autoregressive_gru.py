import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import RankLoss
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from module.transformer_replaced import AutoRegressiveGRU
from utils import get_root_dir, calc_ic

class minGRU(pl.LightningModule):
    def __init__(self, config, n_train_samples):
        super().__init__()
        self.config = config
        self.input_channel = config['vqvae']['input_channel']
        self.num_features = config['vqvae']['num_features']
        self.hidden_size = config['vqvae']['hidden_size']
        self.num_factors = config['vqvae']['num_factors']
        self.dropout = config['vqvae']['dropout']
        self.sos_token_ids = config['vqvae']['num_factors']  # SOS 토큰
        
        # Stage2 모듈: 기존 VQVAE 구조의 encoder, quantizer, decoder는 그대로 사용하고,
        # Stage2 예측 모듈로 AutoRegressiveGRU (즉, AblationGRU 기반)를 사용합니다.
        self.mingru = AutoRegressiveGRU(
            temperature=config['transformer']['temperature'],
            config=config
        )
        
        # CosineAnnealingLR의 T_max (총 step 수)
        self.T_max = config['train']['num_epochs'] * np.ceil(n_train_samples / config['train']['batch_size'] + 1)
        
        self.rank_loss = config['transformer'].get('rank_loss', False)
        if self.rank_loss:
            alpha = config['transformer']['rank_loss_alpha']
            self.mse_loss = RankLoss(alpha=alpha)
        else:
            self.mse_loss = torch.nn.MSELoss()
        
        # 이름 및 로그용 정보 구성 (필요시 수정)
        tf_hidden = config['transformer']['hidden_size']
        tf_head = config['transformer']['heads']
        tf_layers = config['transformer']['n_layers']
        seed = config['train']['seed']
        vq_hidden = config['vqvae']['hidden_size']
        vq_elements = config['vqvae']['num_elements']
        vq_code = config['vqvae']['num_factors']
        self.name = f'AblationGRU_{vq_code}_h{vq_hidden}_e{vq_elements}__Th_{tf_hidden}_h{tf_head}_l{tf_layers}_sd{seed}'
        
        self.ic = []    # IC 기록 리스트
        self.ric = []   # RIC 기록 리스트
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
        Args:
            firm_char: firm-level input (예: (B, ..., feature_dim))
            inputs: (B, T, input_channel) 재구성 대상 데이터
            market: (B, T, market_feature_dim) market 정보
        Returns:
            logits: GRU의 출력 (B, T, vocab_size)
            target: quantizer로부터 산출된 discrete token indices (B, T)
            y_hat: decoder를 통해 재구성한 출력 (B, T, num_elements)
        """
        logits, target, y_hat = self.mingru(firm_char=firm_char, y=inputs, market=market)
        return logits, target, y_hat
    
    def training_step(self, batch, batch_idx):
        if batch.nelement() == 0:
            return None
        
        # 배치 구성: 예시로 firm_char, y, market 정보를 slicing (데이터 포맷에 따라 조정)
        firm_char = batch[:, :, 0:158]            # firm-level 정보
        y = batch[:, :, 158].unsqueeze(-1)          # 재구성 대상 (연속 값)
        market = batch[:, :, 159:]                # market 정보
        
        logits, target, y_hat = self.forward(firm_char, y, market)
        
        # Stage2에서는 quantizer로부터 산출된 discrete target (target indices)에 대해 cross entropy loss를 계산합니다.
        prior_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        mse_loss = self.mse_loss(y_hat, y)
        loss = self.eta * prior_loss + mse_loss
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_prior_loss', prior_loss, prog_bar=True)
        self.log('train_mse_loss', mse_loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        firm_char = batch[:, :, 0:158]
        y = batch[:, :, 158].unsqueeze(-1)
        market = batch[:, :, 159:]
        
        logits, target, y_hat = self.forward(firm_char, y, market)
        
        prior_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-1)
        mse_loss = self.mse_loss(y_hat, y)
        loss = self.eta * prior_loss + mse_loss
        
        self.log('val_loss', loss, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_prior_loss', prior_loss, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_mse_loss', mse_loss, on_epoch=True, logger=True, sync_dist=True)
        
        # 마지막 시점의 예측값과 타깃값으로 IC, RIC 계산 (calc_ic는 외부 함수로 가정)
        daily_ic, daily_ric = calc_ic(
            y_hat[:, -1].squeeze().detach().cpu().numpy(),
            y[:, -1].squeeze().detach().cpu().numpy()
        )
        self.ic.append(daily_ic)
        self.ric.append(daily_ric)
        
        return {"val_loss": loss, "val_prior_loss": prior_loss, "val_mse_loss": mse_loss}
    
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
        
        # 기록 초기화
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