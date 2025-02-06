import time
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import wandb
import os
import gc
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import load_yaml_param_settings, load_args, get_root_dir, seed_everything, run_inference,log_metrics_as_bar_chart
from trainer.autoregressive import minGPT
import logging
from qlib.data.dataset import DatasetH, TSDatasetH, DataHandlerLP
from data.dataset import init_data_loader
torch.set_float32_matmul_precision('high')

def train_stage2():

    #* Load config
    args = load_args()
    config = load_yaml_param_settings(args.config)
    seed_everything(config['train']['seed'])

    #* Init logger
    group_name  = "Stage2"
    project_name = "FactorVQVAE-SWEEP2"
    wandb.init(project=project_name+"-GPT(Sweep)", group= group_name, entity='x7jeon8gi') # ! Name of group
    wandb_config = wandb.config
    wandb_logger = WandbLogger(project=project_name, group= group_name, entity='x7jeon8gi') # ! Name of group

    #* Load dataset
    if config['data']['data_path'].split('.')[-1] == 'csv':
        df = pd.read_csv(config['data']['data_path'])
    elif config['data']['data_path'].split('.')[-1] == 'pkl':
        df = pd.read_pickle(config['data']['data_path'])
    else:
        raise NotImplementedError
    
    handlerlp = DataHandlerLP.from_df(df)

    dic =  {
        'train': config['data']['train_period'],
        'valid': config['data']['valid_period'],
        'test': config['data']['test_period'],
    }

    TsDataset = TSDatasetH(handler = handlerlp, segments=dic, step_len = config['data']['window_size'], fillna_type='ffill+bfill')
    train_prepare = TsDataset.prepare(segments='train', data_key=DataHandlerLP.DK_L)
    valid_prepare = TsDataset.prepare(segments='valid', data_key=DataHandlerLP.DK_L)
    test_prepare = TsDataset.prepare(segments='test', data_key=DataHandlerLP.DK_L)

    train_loader = init_data_loader(train_prepare, shuffle=True, )
    valid_loader = init_data_loader(valid_prepare, shuffle=False, )
    test_loader = init_data_loader(test_prepare, shuffle=False, )

    #! W&B Sweep 파라미터
    config['transformer']['hidden_size'] = wandb_config.hidden_dim
    config['transformer']['rank_loss'] = wandb_config.rank_loss
    config['transformer']['attn_pdrop'] = wandb_config.attn_pdrop
    config['transformer']['n_layers'] = wandb_config.n_layers

    tf_hidden = config['transformer']['hidden_size']
    tf_head = config['transformer']['heads']
    tf_layers = config['transformer']['n_layers']
    seed = config['train']['seed']
    vq_hidden = config['vqvae']['hidden_size']
    vq_elements = config['vqvae']['num_elements']
    vq_code = config['vqvae']['num_factors']
    alpha = config['vqvae']['alpha']
    rank_alpha = config['transformer']['rank_loss_alpha']
    project_name = config['train']['project_name']

    run_name = f'Revise2_a{rank_alpha}_VQ_{vq_code}_h{vq_hidden}_e{vq_elements}__Th_{tf_hidden}_h{tf_head}_l{tf_layers}_sd{seed}' # !Auto

    wandb.run.name = run_name
    wandb.config.update(config)
    
    # * Init model
    n_train_samples = len(train_loader) * config['train']['batch_size'] # approximate
    model = minGPT(config=config, n_train_samples=n_train_samples)
    wandb_logger.watch(model, log='all')

    chekcpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        dirpath=os.path.join(get_root_dir(), 'checkpoints'),
        filename = f'{run_name}'+'-stage2-gpt'+'-{epoch}-{val_loss:.9f}'
    )

    early_stop_callback = EarlyStopping(
        monitor   = 'val_loss',
        min_delta = 0.0001,
        patience  = config['train']['early_stop'], # epochs to wait after min has been reached
        verbose   = True,
        mode      = 'min'
    )

    trainer = pl.Trainer(logger = wandb_logger,
                    enable_checkpointing=True,
                    callbacks=[LearningRateMonitor(logging_interval='step'), chekcpoint_callback, early_stop_callback],
                    max_epochs=config['train']['num_epochs'],
                    accelerator= 'gpu', # 'gpu' # ! 디버깅을 위해 device를 cpu로 설정
                    # strategy='ddp',
                    devices= 1, # config['train']['gpu_counts'] if torch.cuda.is_available() else None,
                    num_nodes=1,
                    precision = config['train']['precision'],
                    )
    
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = valid_loader)

    # Best Model Load
    model = minGPT.load_from_checkpoint(chekcpoint_callback.best_model_path, config=config, n_train_samples=n_train_samples)
    model.eval()

    # run inference
    pred_df, _, metric = run_inference(model, test_loader)
    pred_df.to_pickle(f"{get_root_dir()}/res/{run_name}.pkl")

    # log metric
    log_metrics_as_bar_chart(metric)
    wandb.finish()
    gc.collect()


if __name__ =="__main__":

    #* Train with WandB Sweep
    sweep_config = {
        'method': 'grid', #grid, random
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'hidden_dim': {
                'values': [16, 32, 64, 128, 256]
            },
            'rank_loss': {
                'value': 0.1 # fixed
            },
            'attn_pdrop': {
                'values': [0, 0.1, 0.5]
            },
            'n_layers': {
                'values': [1, 4]
            },
            
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="VQVAE-GPT-Sweep")
    wandb.agent(sweep_id, function=train_stage2)