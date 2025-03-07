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
from utils import UnfreezeDecoderCallback
from trainer.autoregressive import minGPT
import logging
from qlib.data.dataset import DatasetH, TSDatasetH, DataHandlerLP
from data.dataset import init_data_loader
torch.set_float32_matmul_precision('high')

def run_inference_for_checkpoint(checkpoint_path, checkpoint_type, model_class, config, n_train_samples, test_loader, run_name):
    """
    주어진 체크포인트 타입에 따라 모델을 로드하고, inference를 수행하며 결과를 저장하는 함수입니다.
    checkpoint_type은 'loss' 또는 'ric'와 같이 문자열로 전달됩니다.
    """
    # 체크포인트에서 모델 로드 및 평가 모드 전환
    model = model_class.load_from_checkpoint(checkpoint_path, config=config, n_train_samples=n_train_samples)
    model.eval()
    
    # inference 실행
    pred_df, _, metric = run_inference(model, test_loader)
    
    # checkpoint_type을 이용해 결과 저장 경로 지정
    output_path = os.path.join(get_root_dir(), 'res', f"{run_name}_{checkpoint_type}.pkl")
    pred_df.to_pickle(output_path)
    
    # metric을 로깅
    log_metrics_as_bar_chart(metric)
    wandb.log({f'metrics_{checkpoint_type}': metric})
    
    print(f"Inference completed for checkpoint ({checkpoint_type}): {checkpoint_path}")

def train_stage2():

    #* Load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    #* Init logger
    group_name  = "Stage2"
    project_name = "VQVAE2"
    wandb.init(project=project_name+"-GPT(Sweep)", group= group_name)
    wandb_config = wandb.config
    wandb_logger = WandbLogger(project=project_name, group= group_name)
    seed_everything(wandb_config.seed)

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
    config['transformer']['attn_pdrop'] = wandb_config.attn_pdrop
    config['transformer']['n_layers'] = wandb_config.n_layers
    config['vqvae']['num_factors'] = wandb_config.num_factor
    config['transformer']['saved_model'] = wandb_config.saved_model
    config['transformer']['heads'] = wandb_config.n_head
    config['train']['seed'] = wandb_config.seed
    config['transformer']['checkpoint_folder'] = 'checkpoints'
    config['transformer']['rank_loss_alpha'] = wandb_config.rank_loss_alpha
    tf_hidden = config['transformer']['hidden_size']
    tf_head = config['transformer']['heads']
    tf_layers = config['transformer']['n_layers']
    seed = config['train']['seed']
    vq_hidden = config['vqvae']['hidden_size']
    #vq_elements = config['vqvae']['num_elements']
    vq_code = config['vqvae']['num_factors']
    #alpha = config['vqvae']['alpha']
    rank_alpha = config['transformer']['rank_loss_alpha']
    project_name = config['train']['project_name']
    attn_pdrop = config['transformer']['attn_pdrop']
    dec_warmup = config['transformer']['dec_warmup']
    eta = config['transformer']['eta']
    omega = config['transformer']['omega']

    run_name = f'Stage2_VQ{vq_code}_CSI_sd{seed}'

    wandb.run.name = run_name
    wandb.config.update(config)
    
    # * Init model
    n_train_samples = len(train_loader) * config['train']['batch_size'] 
    model = minGPT(config=config, n_train_samples=n_train_samples)
    wandb_logger.watch(model, log='all')
        
    chekcpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        dirpath=os.path.join(get_root_dir(), 'checkpoints_fix'),
        filename = f'{run_name}'+'-{epoch}-{val_loss:.4f}'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta = 0.0001,
        patience = 20,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(logger = wandb_logger,
                    enable_checkpointing=True,
                    callbacks= [LearningRateMonitor(logging_interval='step'), 
                                chekcpoint_callback,
                                early_stop_callback],
                    max_epochs =config['train']['num_epochs'],
                    accelerator = 'gpu', # 'gpu' # ! 디버깅을 위해 device를 cpu로 설정
                    # strategy='ddp',
                    devices   = 1, # config['train']['gpu_counts'] if torch.cuda.is_available() else None,
                    num_nodes = 1,
                    precision = config['train']['precision'],
                    gradient_clip_val = 3.0
                    )
    
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = valid_loader)

    checkpoint_paths = {
        'loss': chekcpoint_callback.best_model_path,
    }

    for ckpt_type, ckpt_path in checkpoint_paths.items():
        run_inference_for_checkpoint(ckpt_path, ckpt_type, minGPT, config, n_train_samples, test_loader, run_name)

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
                'value': 64 # 32 or 64
            },
            'attn_pdrop': {
                'value': 0.5
            },
            'n_head':{
                'value': 2 # 2 or 4
            },
            'n_layers': {
                'value': 2
            },
            'rank_loss_alpha': {
                'value': 0.1
            },

            'saved_model': {
                'value': 'Stage1_VQ_512_CSI.ckpt'
            },
            'num_factor': {
                'value': 512
            },
            'seed': {
                'values': [0,1,2,3,4]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="VQVAE2")
    wandb.agent(sweep_id, function=train_stage2)