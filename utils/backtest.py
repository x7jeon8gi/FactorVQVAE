import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
from pathlib import Path
from module.bidirectional_transformer import BidirectionalTransformer
from module.gpt_transformer import AutoRegressiveTransformer
import math
from data.dataset import DateGroupedBatchSampler, collate_fn, init_data_loader
import pandas as pd 
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from utils import seed_everything
from qlib.data.dataset import DatasetH, TSDatasetH, DataHandlerLP, TSDataSampler

class InvestmentModel:
    def __init__(self, config, model_path=None, seed=42, num_workers=0):
        seed_everything(seed)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model(model_path) if model_path is not None else self.build_model()
        self.model.feature_extractor.stage = 2
        self.model.to(self.device)
        self.model.eval()
        self.num_workers = num_workers
        
    def build_model(self, saved_model_path=None):
        saved_model_path = self.config['transformer']['saved_model'] if saved_model_path is None else saved_model_path
        checkpoint = torch.load(saved_model_path, map_location=self.device)['state_dict']
        
        if self.config['inference']['model'] == "bert":
            model = BidirectionalTransformer(temperature= 1, T =1, config = self.config)
            state_dict = {k.replace('maskgit.', ''): v for k, v in checkpoint.items() if k.startswith('maskgit.')}
        elif self.config['inference']['model'] == "gpt":
            model = AutoRegressiveTransformer(temperature= 1, config = self.config)
            state_dict = {k.replace('mingpt.', ''): v for k, v in checkpoint.items() if k.startswith('mingpt.')}
       
        model.load_state_dict(state_dict)
        
        return model
    
    def build_data_loader(self, data_path):
        print("Load data from {}".format(data_path))

        self.dataframe = pd.read_pickle(data_path)
        handlerlp = DataHandlerLP.from_df(self.dataframe)
        dic = {
            'train' : self.config['data']['train_period'],
            'valid' : self.config['data']['valid_period'],
            'test' : self.config['data']['test_period']
        }
        # todo: check here 
        TsDataset = TSDatasetH(handler = handlerlp, segments = dic, step_len = self.config['data']['window_size'])
        test_prepare = TsDataset.prepare(segments = 'test', data_key=DataHandlerLP.DK_I) # DK_I: inference data # but we have to use DK_L
        test_index = test_prepare.get_index()
        self.dataloader = self.get_dataloader(test_prepare, self.num_workers)

        return test_index
    
    def get_dataloader(self, handler, num_workers):
        dataloader = init_data_loader(handler, shuffle=False, num_workers=num_workers)
        return dataloader
    
    def inference(self, data_path, top_k=3):
        if self.config['inference']['model'] == "bert":
            return self.inference_bert(data_path)

        elif self.config['inference']['model'] == "gpt":
            return self.inference_gpt(data_path, top_k)

    @torch.no_grad()
    def inference_gpt(self, data_path, top_k=3):
        
        test_index = self.build_data_loader(data_path)
        pred = []
        real = []
        loss = []
        z_e_list = []
        print("Start inference")
        for batch in tqdm(self.dataloader):
        
            firm_char = batch[:, : ,0:158].to(self.device)
            y = batch[:, :, 158].unsqueeze(-1).to(self.device)
            market = batch[:, :, 159:].to(self.device)
            
            # feature extraction
            firm_char = self.model.feature_extractor(firm_char) 
            # firm_char_t_2 = firm_char[:,:-1,:] #* x_(t-2)
            # firm_char_t_1 = firm_char[:, -1,:] #* x_(t-1)

            market = self.model.market_extractor(market) # market은 transformer에 모두 사용
            inputs_t_1 = y[:,:-1,:] #* y_(t-1):: t-1 까지의 스텝
            y = y[:,-1,:] #* y_t (ground truth):: 한 시점 다음 스텝 예측

            # encode to z_q
            z_e = self.model.encoder(inputs_t_1)
            z_q , vq_dict = self.model.quantizer(z_e)
            idx = vq_dict['q'].squeeze() # transformer input

            # Add sos token in the first position
            sos_token = torch.ones((idx.size(0), 1, ), dtype=torch.long) * self.model.sos_token_ids
            sos_token = sos_token.long().to(self.device)
            idx = torch.cat([sos_token, idx], dim=1).long()

            # transformer
            if self.config['transformer']['use_market']:
                market = self.mingpt.market_extractor(market)
                logits = self.mingpt.transformer(idx, market)

            else:
                logits = self.mingpt.transformer(idx)
            logit = logits[:, -1, :] # get last token logit :: y^_t
            
            if top_k is not None:
                logit = self.top_k_logits(logit, top_k)
            # probs = F.softmax(logit, dim=-1)
            # ix = torch.multinomial(probs, num_samples=1)
            ix = torch.argmax(logit, dim=-1).unsqueeze(-1)
            sampling_idx = torch.cat([idx, ix], dim=1) 
            # get rid of sos token
            sampling_idx = sampling_idx[:, 1:]

            # get quantized value from codebook (B x N x C)
            quantize = F.embedding(sampling_idx, self.model.quantizer.get_codebook().to(self.device)) 
            # get decoder output
            y_hat, _ = self.model.decoder(firm_char = firm_char, inputs = quantize)
            y_hat = y_hat[:, -1, :]
            reconstr_loss = F.mse_loss(y_hat, y)

            pred.append(y_hat.cpu().detach().numpy())
            real.append(y.cpu().detach().numpy())
            loss.append(reconstr_loss.cpu().detach().numpy())
            z_e_list.append(z_e.cpu().detach().numpy())

        pred = pd.Series(np.concatenate(pred, axis=0).squeeze(), index=test_index)
        real = pd.Series(np.concatenate(real, axis=0).squeeze(), index=test_index)
        loss = np.mean(loss)

        return pred, real, loss, z_e_list

    @torch.no_grad()
    def check_tokenizer(self, data_path):
        print("Load data from {}".format(data_path))
        self.dataframe = pd.read_csv(data_path)
        self.dataloader = self.get_dataloader(self.dataframe)
        print("Start inference")
        idx_list = []
        for batch in tqdm(self.dataloader):
            
            firm_char, inputs = batch # [batch_size, window_size, feature_dim]  #* x_(t-2)
            firm_char = firm_char.to(self.device)
            inputs = inputs.to(self.device)

            firm_char_t_2 = self.model.feature_extractor(firm_char[:,:-1,:]) #* x_(t-2)
            # firm_cahr_t_1 = self.model.feature_extractor(firm_char)[:,-1,:] #* x_(t-1)
            inputs_t_1 = inputs[:,:-1,:] #* y_(t-1)
            y = inputs[:,-1,:] #* y_t (ground truth)
            
            z_e = self.model.encoder(firm_char_t_2, inputs_t_1)
            z_q, vq_dict = self.model.quantizer(z_e)
            idx = vq_dict['q'].squeeze()
            idx_list.append(idx.cpu().detach().numpy())

        return idx_list

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out
    

    @torch.no_grad()
    def inference_bert(self, data_path):
        test_index = self.build_data_loader(data_path)

        pred = []
        real = []
        loss = []
        print("Start inference")
        for batch in tqdm(self.dataloader):
            
            firm_char = batch[:, : ,0:158]
            inputs = batch[:, :, 158].unsqueeze(-1)
            market = batch[:, :, 159:] # no use in bert

            firm_char = firm_char.to(self.device)
            inputs = inputs.to(self.device)

            firm_char_t_2 = self.model.feature_extractor(firm_char[:,:-1,:]) #* x_(t-2)
            firm_char_t_1 = self.model.feature_extractor(firm_char[:,-1,:]) #* x_(t-1)
            inputs_t_1 = inputs[:,:-1,:] #* y_(t-1)
            y = inputs[:,-1,:] #* y_t (ground truth)
            
            z_e = self.model.encoder(inputs_t_1)
            z_q, vq_dict = self.model.quantizer(z_e)
            idx = vq_dict['q'].squeeze()

            # Add Mask token in the last position
            mask_token = torch.ones((idx.size(0), 1, )).to(idx.device) * self.model.mask_token_ids
            idx = torch.cat([idx, mask_token], dim=1).long() # * get y_t

            logit = self.model.transformer(idx)
            # ?categorical distribution or multinomial distribution ?
            # sample_ids = torch.distributions.Categorical(logits=logit).sample()
            # sample_ids = torch.multinomial(F.softmax(logit, dim=-1), num_samples=1).squeeze()
            sample_ids = torch.argmax(logit, dim=-1).squeeze()

            quantize = F.embedding(sample_ids, self.model.quantizer.get_codebook().to(self.device))
            quantize_t_1 = quantize[:,-1,:] #* y_(t-1)

            # ! New model: we need hidden state from previous step.
            _, hidden_states = self.model.decoder(firm_char=firm_char_t_2 , inputs = quantize[:,:-1,:])
            y_hat, _ = self.model.decoder(firm_char = firm_char_t_1.unsqueeze(1) , inputs=quantize_t_1.unsqueeze(1), hidden = hidden_states)
            y_hat = y_hat.squeeze(1)
            
            reconstr_loss = F.mse_loss(y_hat, y)

            pred.append(y_hat.cpu().detach().numpy())
            real.append(y.cpu().detach().numpy())
            loss.append(reconstr_loss.cpu().detach().numpy())
            
        pred = pd.Series(np.concatenate(pred, axis=0).squeeze(), index=test_index)
        real = pd.Series(np.concatenate(real, axis=0).squeeze(), index=test_index)
        loss = np.mean(loss)

        return pred, real, loss
    

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
        'IC_IR': np.mean(ric) / np.std(ric),
        'RankIC': np.mean(ric),
        'RankIC_IR': np.mean(ric) / np.std(ric)
    }

    return pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    


def calculate_table_metrics(series, period, name, target_return=0):

    if period is not None:
        if type(period) == int:
            series = series[series.index.year == int(period)].copy()
            # series['return'] = series['return'] / series['return'].iloc[0]  
        elif type(period) == list:
            series = series.loc[period[0]:period[1]].copy()
    try:  
        daily_log_returns = series['return']
        cum_return = series['return'].cumsum()
    except:
        daily_log_returns = series
        cum_return = series.cumsum()
    normal_cum_return = np.exp(cum_return)
    
    # MDD 계산을 위해 누적 일반 리턴 사용
    max_cumulative_returns = normal_cum_return.cummax()
    drawdown = (normal_cum_return - max_cumulative_returns) / (max_cumulative_returns + 1e-9) 
    mdd = drawdown.min()

    # 연간 수익률 및 기타 지표 계산
    annual_return = daily_log_returns.mean() * 252
    annual_std = daily_log_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_std

    # Sortino Ratio
    # Calculate downside deviation
    downside_returns = daily_log_returns[daily_log_returns < target_return]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - target_return) / downside_std if downside_std != 0 else np.nan
    
    # Calmar Ratio
    calmar_ratio = annual_return / abs(mdd) if mdd != 0 else np.nan

    # Turnover
    turnover = series['turnover'].mean()
    turnover = round(turnover, 4)
    
    result = {
        'Annualized Return': round(annual_return, 4),
        'Annual Std': round(annual_std, 4),
        'Sharpe Ratio': round(sharpe_ratio, 4),
        'Sortino Ratio': round(sortino_ratio, 4),
        'Calmar Ratio': round(calmar_ratio, 4),
        'MDD': round(mdd, 4),
        'Cumulative Returns': round(cum_return.iloc[-1], 4),
        'Turnover': turnover
    }

    return pd.DataFrame.from_dict(result, orient='index', columns=[f'{name}'])