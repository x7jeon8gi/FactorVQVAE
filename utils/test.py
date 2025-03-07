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
        'ICIR': np.mean(ic) / np.std(ic),
        'RankIC': np.mean(ric),
        'RankICIR': np.mean(ric) / np.std(ric)
    }

    return metrics
    # return pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])

@torch.no_grad()
def run_inference_cont(model,data_loader,device='cuda'):
    model.eval()
    model.to(device)
    preds = []
    reals = []
    
    # test index를 얻는 함수는 사용자가 구현한 것으로 가정합니다.
    test_index = data_loader.dataset.get_index()
    
    # 배치 단위 inference loop
    for batch in tqdm(data_loader, desc="Running Inference GRU Iterative"):
        batch = batch.to(device)
        # 배치에서 각 입력 구성 요소 추출
        firm_char = batch[:, :, 0:158]              # firm-level 정보
        inputs = batch[:, :, 158].unsqueeze(-1)       # 재구성 대상 시퀀스 (토큰으로 표현된 값)
        market = batch[:, :, 159:]                    # market 정보

        firm_features = model.cont_mingpt.feature_extractor(firm_char)

        # autoregressive inference를 위해 context 토큰만 사용
        # inputs_t_1: context 토큰 (마지막 토큰은 정답으로 분리)
        inputs_t_1 = inputs[:, :-1, :]
        target_token = inputs[:, -1, :]  # 정답 토큰 (평가용)

        z_mu, z_logvar = model.cont_mingpt.encode(inputs_t_1)
        z = model.cont_mingpt.reparameterize(z_mu, z_logvar)
        z = model.cont_mingpt.prepare_inputs(z, device)

        if model.config['transformer']['use_market']:
            market_features = model.cont_mingpt.market_extractor(market)
            z_transformed = model.cont_mingpt.latent_transformer(z, market=market_features)
        else:
            z_transformed = model.cont_mingpt.latent_transformer(z)

        z_transformed = model.cont_mingpt.linear(z_transformed)
        y_hat, _ = model.cont_mingpt.decoder(firm_char=firm_features, inputs=z_transformed)

        generated = y_hat[:, -1, :]
        preds.append(generated.cpu().detach().numpy())
        reals.append(target_token.cpu().detach().numpy())

    preds = pd.Series(np.concatenate(preds, axis=0).squeeze(), index=test_index)
    reals = pd.Series(np.concatenate(reals, axis=0).squeeze(), index=test_index)
    df = pd.DataFrame({'score': preds, 'label': reals})
    
    # 평가 지표 (RankIC, IC/IR) 계산: 해당 함수들은 사용자가 정의한 것으로 가정합니다.
    rankic = RankIC(df, column1='score', column2='label')
    print(f"RankIC: {rankic}")
    icir = Cal_IC_IR(df, column1='label', column2='score')
    print(f"Metrics: {icir}")
    
    return df, rankic, icir



@torch.no_grad()
def run_inference_gru(model, data_loader, device='cuda'):
    model.eval()
    model.to(device)
    preds = []
    reals = []
    
    # test index를 얻는 함수는 사용자가 구현한 것으로 가정합니다.
    test_index = data_loader.dataset.get_index()
    
    # 배치 단위 inference loop
    for batch in tqdm(data_loader, desc="Running Inference GRU Iterative"):
        batch = batch.to(device)
        # 배치에서 각 입력 구성 요소 추출
        firm_char = batch[:, :, 0:158]              # firm-level 정보
        inputs = batch[:, :, 158].unsqueeze(-1)       # 재구성 대상 시퀀스 (토큰으로 표현된 값)
        market = batch[:, :, 159:]                    # market 정보

        # autoregressive inference를 위해 context 토큰만 사용
        # inputs_t_1: context 토큰 (마지막 토큰은 정답으로 분리)
        inputs_t_1 = inputs[:, :-1, :]
        target_token = inputs[:, -1, :]  # 정답 토큰 (평가용)

        # VQ encoder/quantizer를 통해 context에 해당하는 discrete token indices 생성
        # (여기서 y로 context inputs만 사용)
        _, target_indices = model.mingru.encode_to_z_q(inputs_t_1)
        # prepare_inputs()는 target_indices에 SOS 토큰을 prepend하여 input_indices를 생성합니다.
        # shape: (B, L_context+1)
        input_indices = model.mingru.prepare_gru_inputs(target_indices, device)

        # market feature는 한번만 추출 (autoregressive loop 동안 동일)
        market_features = model.mingru.market_extractor(market) if model.config['transformer']['use_market'] else None

        # 초기 생성 시퀀스: context (이미 SOS 토큰이 포함됨)
        generated = input_indices  # shape: (B, L_context+1)


        if model.config['transformer']['use_market']:
            logits = model.mingru.sequential(generated, market_features=market_features)
        else:
            logits = model.mingru.sequential(generated)
        # logits의 마지막 시점에서 다음 토큰 예측 (argmax)
        next_logits = logits[:, -1, :]  # (B, vocab_size)
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # (B, 1)
        # 예측된 토큰을 생성 시퀀스에 append
        generated = torch.cat([generated, next_token], dim=1)

        # 생성된 시퀀스에서 SOS 토큰을 제거하여 최종 예측 token sequence로 사용
        predicted_indices = generated[:, 1:]  # (B, L_context + n_steps)

        # firm-level feature 추출
        firm_features = model.mingru.feature_extractor(firm_char)
        # Decoder: 예측된 token sequence를 코드북 임베딩 lookup 후 firm feature와 함께 재구성 수행
        y_hat = model.mingru.decode_quantized_embeddings(firm_features, predicted_indices)
        # 최종 예측: 재구성된 결과의 마지막 시점의 값 사용 (autoregressive 마지막 토큰)
        y_pred = y_hat[:, -1, :]

        preds.append(y_pred.cpu().detach().numpy())
        reals.append(target_token.cpu().detach().numpy())
    
    preds = pd.Series(np.concatenate(preds, axis=0).squeeze(), index=test_index)
    reals = pd.Series(np.concatenate(reals, axis=0).squeeze(), index=test_index)
    df = pd.DataFrame({'score': preds, 'label': reals})
    
    # 평가 지표 (RankIC, IC/IR) 계산: 해당 함수들은 사용자가 정의한 것으로 가정합니다.
    rankic = RankIC(df, column1='score', column2='label')
    print(f"RankIC: {rankic}")
    icir = Cal_IC_IR(df, column1='label', column2='score')
    print(f"Metrics: {icir}")
    
    return df, rankic, icir

@torch.no_grad()
def run_inference(model, data_loader, device='cuda'):

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


@torch.no_grad()
def run_interpret(model, data_loader, device='cuda'):
    """
    시간이 변함에 따라 실제로 어떤 codebook index가 생성되는지 확인하는 함수
    이를 시각화 하는 것이 목표임
    """
    model.eval()
    model.to(device)

    codebook_ls = []
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

        sos_token = torch.ones((1, ), dtype=torch.long) * model.mingpt.sos_token_ids
        sos_token = sos_token.long().to(device)
        idx = torch.cat([sos_token, idx], dim=0).long()
        idx =idx.unsqueeze(0)

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


        #!## codebook index의 분포 계산
        #! idx의 분포 계산: 각 고유 값이 몇 번 등장하는지
        idx_dist = sampling_idx.cpu().detach().numpy()#[:,-1] #[:, -1] # 마지막 2개의 시점에 대한 분포만 계산
        idx_dist = idx_dist.reshape(-1)
        values, counts = np.unique(idx_dist, return_counts=True)
        result = dict(zip(values, counts))
        
        # 리스트에 결과 저장 (배치별)
        codebook_ls.append(result)
    
    codebook_df = pd.DataFrame(codebook_ls, index=test_index.get_level_values(0).unique()).fillna(0)
    codebook_df.to_pickle('temp/codebook.pkl')

    return codebook_df