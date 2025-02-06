import os
import gc
import torch
import numpy as np
import wandb
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from trainer.autoencoder import FactorVQVAE
from qlib.data.dataset import TSDatasetH, DataHandlerLP
from utils import load_yaml_param_settings, load_args, get_root_dir, seed_everything
from data.dataset import init_data_loader
torch.set_float32_matmul_precision('high')

def train():

    args = load_args()
    config = load_yaml_param_settings(args.config)
    
    #* Init logger
    group_name = "Stage1"
    project_name = "FactorVQVAE-SWEEP1"
    wandb.init(project=project_name,  group= group_name, entity="x7jeon8gi") # todo: group_name
    wandb_config = wandb.config
    wandb_logger = WandbLogger(project=project_name, entity="x7jeon8gi") # todo: group_name

    config['train']['seed'] = wandb_config.seed
    seed_everything(config['train']['seed'])

    #* Load dataset
    if config['data']['data_path'].split('.')[-1] == 'csv':
        df = pd.read_csv(config['data']['data_path'])
    elif config['data']['data_path'].split('.')[-1] == 'pkl':
        df = pd.read_pickle(config['data']['data_path']).astype(float)
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

    train_loader = init_data_loader(train_prepare, shuffle=True)
    valid_loader = init_data_loader(valid_prepare, shuffle=False)

    #* Init model
    n_train_samples = len(train_loader.dataset)

    #! W&B Sweep 파라미터
    # config['vqvae']['hidden_size'] = wandb_config.hidden_dim
    config['vqvae']['num_factors'] = wandb_config.num_factors
    # config['vqvae']['num_heads'] = wandb_config.num_heads
    codebook_sizes = config['vqvae']['num_factors']
    hidden_size = config['vqvae']['hidden_size']
    elements = config['vqvae']['num_elements']
    alpha = config['vqvae']['alpha']
    project_name = config['train']['project_name']
    seed = config['train']['seed']
    if config['train']['run_name'] is not None:
        run_name = f'Revise_VQ1_C{codebook_sizes}_h{hidden_size}_e{elements}_sd{seed}' # !Auto
    else:
        run_name = None

    wandb.run.name = run_name
    wandb.config.update(config)

    # 호환되지 않는 조합을 검사하고 처리
    # if config['vqvae']['hidden_size'] % config['vqvae']['num_heads'] != 0:
    #     print(f"Skipping incompatible combination: hidden_dim={wandb_config.hidden_dim}, num_heads={wandb_config.num_heads}")
    #     return  # 이 실험을 스킵하고 다음 조합으로 넘어감

    #* Init model
    n_train_samples = len(train_loader) * config['train']['batch_size'] # approximate
    model = FactorVQVAE(config, n_train_samples, ckpt_path=None, ignore_keys=list())
    wandb_logger.watch(model, log='all')

    #* Init Trainer
    chekcpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        dirpath=os.path.join(get_root_dir(), 'checkpoints'),
        filename = f'{run_name}'+'-{epoch}-{val_loss:.5f}'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10, # epochs to wait after min has been reached
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(logger = wandb_logger,
                         enable_checkpointing=True,
                         callbacks=[LearningRateMonitor(logging_interval='step'), 
                                    chekcpoint_callback, 
                                    early_stop_callback],
                         max_epochs=config['train']['num_epochs'],
                         accelerator= 'gpu', # 'gpu'
                         # strategy='ddp',
                         devices= 1, # config['train']['gpu_counts'] if torch.cuda.is_available() else None,
                         precision = config['train']['precision'],
                         )

    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = valid_loader)

    wandb.finish()
    gc.collect()

if __name__ == "__main__":


    #* Train with WandB Sweep
    sweep_config = {
        'method': 'grid', #grid, random
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'num_factors': {
                'values': [256,512,1024]
            },
            'seed':{
                'value': 2
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="FactorVQVAE-SWEEP1",entity="x7jeon8gi")
    wandb.agent(sweep_id, function=train)