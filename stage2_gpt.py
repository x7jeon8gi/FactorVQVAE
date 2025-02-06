import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
from typing import Union, Callable, Optional
import wandb
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import load_yaml_param_settings, load_args, get_root_dir, save_model, seed_everything, run_inference
from trainer.autoregressive import minGPT
import logging
from qlib.data.dataset import DatasetH, TSDatasetH, DataHandlerLP
from data.dataset import init_data_loader
import time
torch.set_float32_matmul_precision('high')

def check_and_update_group_name(config):
    if config['train']['group_name'] == 'BERT':
        print("Warning: You are using GPT pretrained model. Please check the config file.")
        config['train']['group_name'] = 'GPT'
        config['train']['run_name'] = config['train']['run_name'].replace('BERT', 'GPT')

def update_config(config, key1, key2, key3):
    if config[key1][key2] != config['transformer'][key3]:
        config[key1][key2] = config['transformer'][key3]

def train_stage2(config, train_loader, valid_loader, test_loader):
    
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
    if config['train']['run_name'] is not None:
        run_name = f'Revise2_a{rank_alpha}_VQ{vq_code}_h{vq_hidden}_e{vq_elements}_Th{tf_hidden}_h{tf_head}_l{tf_layers}_sd{seed}' # !Auto
    else:
        raise NotImplementedError("run_name should be specified. We recommend to use the same run_name as stage1.")

    # * Init model
    n_train_samples = len(train_loader) * config['train']['batch_size'] # approximate

    model = minGPT(config=config, n_train_samples=n_train_samples)
    
    #* init logger
    group_name = config['train']['group_name'] if config['train']['group_name'] is not None else "실험 중"
    wandb.init(project=project_name+'-2-GPT', name=run_name, config=config, group= group_name,entity="x7jeon8gi") # todo: group_name
    wandb_logger = WandbLogger(project=project_name, name=run_name, config=config,entity="x7jeon8gi")
    wandb_logger.watch(model, log='all')

    chekcpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        dirpath=os.path.join(get_root_dir(), 'checkpoints'),
        filename = f'{run_name}'+'-GPT'+'-{epoch}-{val_loss:.4f}'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=50, # epochs to wait after min has been reached
        verbose=True,
        mode='min'
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
    pred_df, rank_ic, metric = run_inference(model, test_loader)
    pred_df.to_pickle(f"{get_root_dir()}/res/{run_name}.pkl")

    # log metric
    wandb.log(metric)

    logging.info("Saving Models.")
    save_model({'maskgit': model.mingpt})

    wandb.finish()

if __name__ =="__main__":
    #* Load config
    args = load_args()
    config = load_yaml_param_settings(args.config)
    seed_everything(config['train']['seed'])

    update_config(config, 'data', 'window_size', 'num_tokens')
    # update_config(config, 'vqvae', 'num_factors', 'codebook_sizes')
    check_and_update_group_name(config)

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

    train_loader = init_data_loader(train_prepare, shuffle=True, )#batch_size=config['train']['batch_size'])
    valid_loader = init_data_loader(valid_prepare, shuffle=False, )#batch_size=config['train']['batch_size'])
    test_loader = init_data_loader(test_prepare, shuffle=False, )#batch_size=config['train']['batch_size'])

    #* Train
    start_time = time.time()
    train_stage2(config, train_loader, valid_loader, test_loader)
    end_time = time.time()

    logging.info(f"Training time: {end_time - start_time} seconds.")