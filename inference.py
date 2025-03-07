import pandas as pd
import torch
import os
from utils import load_yaml_param_settings, load_args, run_inference, seed_everything, get_root_dir
from trainer.autoregressive import minGPT
from qlib.data.dataset import TSDatasetH, DataHandlerLP
from data.dataset import init_data_loader
import tqdm

def main(ckpt, config, test_loader):

    model = minGPT(config, n_train_samples=0)
    model = model.load_from_checkpoint(ckpt)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    codebook_df, rankic, icir = run_inference(model, test_loader, device=device)

    return codebook_df


if __name__ == "__main__":

    args = load_args()
    config = load_yaml_param_settings(args.config)
    seed_everything(config['train']['seed'])

    all_checkpoints = [
        'Stage2_VQ512_CSI_sd0.ckpt',
        'Stage2_VQ512_CSI_sd1.ckpt',
        'Stage2_VQ512_CSI_sd2.ckpt',
        'Stage2_VQ512_CSI_sd3.ckpt',
        'Stage2_VQ512_CSI_sd4.ckpt'
    ]

    df = pd.read_pickle(config['data']['data_path'])
    handlerlp = DataHandlerLP.from_df(df)
    segments = {
        'train': config['data']['train_period'],
        'valid': config['data']['valid_period'],
        'test': config['data']['test_period'],
    }
    TsDataset = TSDatasetH(
        handler=handlerlp, 
        segments=segments, 
        step_len=config['data']['window_size'], 
        fillna_type='ffill+bfill'
    )
    test_prepare = TsDataset.prepare(segments='test', data_key=DataHandlerLP.DK_L)
    test_loader = init_data_loader(test_prepare, shuffle=False)

    for checkpoint in all_checkpoints:
        checkpoint_path = os.path.join(get_root_dir(), 'checkpoints', checkpoint)
        codebook_df = main(checkpoint_path, config, test_loader)
        #print(codebook_df)
