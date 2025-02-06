import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import spearmanr

def RankIC(df, column1='LABEL0', column2='Pred'):
    ric_values_multiindex = []

    for date in df.index.get_level_values(0).unique():
        daily_data = df.loc[date].copy()
        daily_data['LABEL0_rank'] = daily_data[column1].rank()
        daily_data['pred_rank'] = daily_data[column2].rank()
        ric, _ = spearmanr(daily_data['LABEL0_rank'], daily_data['pred_rank'])
        ric_values_multiindex.append(ric)

    if not ric_values_multiindex:
        return np.nan, np.nan

    ric = np.mean(ric_values_multiindex)
    std = np.std(ric_values_multiindex)
    ir = ric / std if std != 0 else np.nan
    return pd.DataFrame({'RankIC': [ric], 'RankIC_IR': [ir]})

def calc_ic(pred, label):
    df = pd.DataFrame({'pred': pred, 'label': label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def Cal_IC_IR(df, column1='LABEL0', column2='Pred'):
    ic = []
    ric = []

    for date in df.index.get_level_values(0).unique():
        daily_data = df.loc[date].copy()
        daily_data['LABEL0'] = daily_data[column1]
        daily_data['pred'] = daily_data[column2]
        ic_, ric_ = calc_ic(daily_data['pred'], daily_data['LABEL0'])
        ic.append(ic_)
        ric.append(ric_)

    metrics = {
        'IC': np.mean(ic),
        'ICIR': np.mean(ric) / np.std(ric),
        'RankIC': np.mean(ric),
        'RankICIR': np.mean(ric) / np.std(ric)
    }

    return metrics
    # return pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    

@torch.no_grad()
def  run_inference(model, data_loader, device='cuda'):

    model.eval()
    model.to(device)
    preds = []
    reals = []

    test_index = data_loader.dataset.get_index()

    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Running Inference")):
        batch = batch.to(device)

        firm_char = batch[:, :, 0:158]
        inputs = batch[:, :, 158].unsqueeze(-1)
        market = batch[:, :, 159:]

        firm_char     = model.mingpt.feature_extractor(firm_char) 
        # firm_char_t = firm_char[:, :-1, :] # (B, T-1, 1)
        # firm_chat_t_1 = firm_char[:, -1, :] # (B, 1, 1)

        inputs_t_1 = inputs[:, :-1, :] # (B, T-1, 1)
        y = inputs[:, -1, :] # (B, 1, 1)

        z_e = model.mingpt.encoder(inputs_t_1)
        z_q, vq_dict = model.mingpt.quantizer(z_e) # 
        idx = vq_dict['q'].squeeze()

        sos_token = torch.ones((idx.size(0), 1, ), dtype=torch.long) * model.mingpt.sos_token_ids
        sos_token = sos_token.long().to(device)
        idx = torch.cat([sos_token, idx], dim=1).long()

        # market feature 사용 여부
        if model.config['transformer']['use_market']:
            market_feat = model.mingpt.market_extractor(market)
            logits = model.mingpt.transformer(idx, market_feat)
        else:
            logits = model.mingpt.transformer(idx)
        logit = logits[:, -1, :]

        # probs = F.softmax(logit, dim=-1)
        # ix = torch.multinomial(probs, num_samples=1)
        
        ix = torch.argmax(logit, dim=-1).unsqueeze(-1)
        sampling_idx = torch.cat([idx, ix], dim=1) # (B, T+1)
        # get rid of sos token
        sampling_idx = sampling_idx[:, 1:] # (B, T)

        # get quantized value from codebook (B x N x C)
        quantize = F.embedding(sampling_idx, model.mingpt.quantizer.get_codebook().to(device)) 
        # get decoder output
        y_hat, _ = model.mingpt.decoder(firm_char = firm_char, inputs = quantize)
        y_hat = y_hat[:,-1,:]

        preds.append(y_hat.cpu().detach().numpy())
        reals.append(y.cpu().detach().numpy())

    preds = pd.Series(np.concatenate(preds, axis=0).squeeze(), index=test_index)
    reals = pd.Series(np.concatenate(reals, axis=0).squeeze(), index=test_index)
    df = pd.DataFrame({'score': preds, 'label': reals})

    rankic = RankIC(df, column1='score', column2='label')
    print(f"RankIC: {rankic}")
    icir = Cal_IC_IR(df, column1='label', column2='score')
    print(f"Metrics: {icir}")

    return df, rankic, icir