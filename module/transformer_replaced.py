import torch
import torch.nn as nn
import torch.nn.functional as F
from module.mingpt import GPT
import copy
import numpy as np
from pathlib import Path
from module.vqvae import FactorEncoder, FactorDecoder, FeatureExtractor
from vqtorch.nn import VectorQuant
import os
import sys
import math 
from utils import freeze, get_root_dir, load_pretrained_tok_emb

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

class AblationGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # vocab_size는 codebook의 크기와 동일 (VQ에서 사용한 num_factors)
        self.vocab_size = config['vqvae']['num_factors']
        self.hidden_size = config['transformer']['hidden_size']  # GRU의 hidden dimension
        self.pkeep = config['transformer']['pkeep']
        self.num_tokens = config['transformer']['num_tokens'] + 1  # SOS 토큰 포함
        
        # Token embedding: vocab_size+1 (SOS 포함) x hidden_size
        self.tok_emb = nn.Embedding(self.vocab_size + 1, self.hidden_size)
        # Positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, self.hidden_size))
        self.dropout = nn.Dropout(config['transformer']['attn_pdrop'])
        # GRU: config['transformer']['n_layers']개의 layer 사용 (batch_first=True)
        self.num_layers = config['transformer'].get('n_layers', 1)
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=True)
        # 최종 출력을 vocab 차원으로 매핑
        self.head = nn.Linear(self.hidden_size, self.vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
    
    def prepare_inputs(self, target_indices, device):
        """
        target_indices: (B, T) tensor of discrete indices from VQ
        Returns: input_indices of shape (B, T+1), with SOS token prepended and masking applied.
        """
        # SOS token: 코드는 보통 VQ codebook의 마지막 index 또는 config에 정의된 값
        sos_tokens = torch.full((target_indices.shape[0], 1), self.config['vqvae']['num_factors'],
                                  dtype=torch.long, device=device)
        
        # Optional masking for denoising training (pkeep < 1.0이면 일부 토큰을 랜덤 교체)
        if self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep * torch.ones(target_indices.shape, device=device)).round().to(torch.int64)
            random_indices = torch.randint_like(target_indices, self.vocab_size, device=device)
            masked_indices = mask * target_indices + (1 - mask) * random_indices
        else:
            masked_indices = target_indices
        # Concatenate SOS 토큰과 masked token sequence
        input_indices = torch.cat((sos_tokens, masked_indices), dim=1)
        return input_indices

    def forward(self, input_indices, market_features=None):
        """
        Args:
            input_indices: (B, T) discrete token indices (SOS 포함)
            market_features: (Optional) (B, T, hidden_size) – market 정보를 투영한 feature
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_indices.shape
        # Token 임베딩
        x = self.tok_emb(input_indices)  # (B, T, hidden_size)
        # Positional embedding (T가 pos_emb의 길이보다 작거나 같다고 가정)
        x = x + self.pos_emb[:, :T, :]
        x = self.dropout(x)
        # 만약 market feature가 있다면 elementwise 합산
        if market_features is not None:
            # market_features의 shape는 (B, T, hidden_size)여야 함.
            x = x + market_features
        # GRU 처리 (batch_first=True)
        output, _ = self.gru(x)  # output: (B, T, hidden_size)
        # 최종 linear layer로 logits 산출
        logits = self.head(output)  # (B, T, vocab_size)
        return logits


# AutoRegressiveGRU: VQ-VAE Stage2에서 Transformer 대신 GRU로 예측
class AutoRegressiveGRU(nn.Module):
    """
    Ablation study:
      기존 Transformer를 간단한 GRU 기반 모델로 대체한 Stage2 구조.
      VQ를 유지하며, Encoder, Quantizer, Decoder는 동일하게 사용하고,
      GRU를 통해 discrete latent code sequence를 예측합니다.
    """
    def __init__(self, temperature, config):
        super().__init__()
        self.sos_token_ids = config['vqvae']['num_factors']  # SOS 토큰 값 (보통 codebook 크기와 동일)
        self.config = config
        self.num_factors = config['vqvae']['num_factors']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 기본 설정: hidden dimension, dropout, 등
        self.dim = config['vqvae']['hidden_size']
        self.input_channel = config['vqvae']['input_channel']
        self.dropout = config['vqvae']['dropout']
        self.num_heads = config['vqvae']['num_heads']
        self.num_features = config['vqvae']['num_features']
        
        # Feature extractor (firm-level 정보를 투영)
        self.feature_extractor = FeatureExtractor(
            num_latent=self.num_features,
            hidden_size=self.dim
        )
        
        # Encoder: 연속적인 입력을 처리하여 latent representation 산출
        self.encoder = FactorEncoder(
            input_size=self.input_channel,
            hidden_size=self.dim,
            num_heads=self.num_heads,
            use_attn=True,
            dropout=self.dropout
        )
        
        # Decoder: firm-level feature와 quantized embedding을 받아 재구성 수행
        self.decoder = FactorDecoder(
            input_size=self.dim, hidden_size=self.dim,
            num_elements=config['vqvae']['num_elements']
        )
        
        # Quantizer: VQ 모듈 (코드북 관련)
        self.quantizer = VectorQuant(
            feature_size=self.dim,                                 # feature dimension
            num_codes=self.num_factors,                            # number of codebook vectors
            beta=self.config['quantizer']['beta'],                 # commitment trade-off
            kmeans_init=self.config['quantizer']['kmeans_init'],   # kmeans++ 초기화 여부
            norm=None,
            cb_norm=None,
            affine_lr=self.config['quantizer']['affine_lr'],
            sync_nu=self.config['quantizer']['sync_nu'],
            replace_freq=self.config['quantizer']['replace_freq'],
            dim=-1
        )
        
        # load pretrained 모델 (feature_extractor, encoder, decoder, quantizer)
        self.load_pretrained_model(config)
        
        # GRU 기반 sequence 모델: AblationGRU를 사용
        self.vocab_size = self.config['vqvae']['num_factors']
        self.pkeep = self.config['transformer']['pkeep']
        self.sequential = AblationGRU(config)
        
        # Market extractor: market 정보를 추출하여 GRU 입력에 합산
        self.use_market = config['transformer']['use_market']
        self.market_extractor = FeatureExtractor(
            num_latent=config['vqvae']['market_features'],
            hidden_size=config['transformer']['hidden_size']
        )
    
    def load_pretrained_model(self, config):
        from pathlib import Path
        # get_root_dir()는 프로젝트 루트 디렉토리를 반환하는 사용자 정의 함수라고 가정합니다.
        saved_model = config['transformer']['saved_model']
        saved_model = f"{saved_model}.ckpt" if saved_model and not saved_model.endswith('.ckpt') else saved_model
        checkpoint_path = Path(get_root_dir()).joinpath('checkpoints_fix', saved_model)
        checkpoint = torch.load(checkpoint_path)['state_dict']
        
        def load_state_dict(module, prefix):
            state_dict = {k.replace(f'{prefix}.', ''): v for k, v in checkpoint.items() if k.startswith(prefix)}
            module.load_state_dict(state_dict)
        
        load_state_dict(self.feature_extractor, 'feature_extractor')
        load_state_dict(self.encoder, 'encoder')
        load_state_dict(self.decoder, 'decoder')
        load_state_dict(self.quantizer, 'quantizer')
        
        # freeze: 파라미터 업데이트를 막는 사용자 정의 함수라고 가정합니다.
        freeze(self.encoder)
        freeze(self.quantizer)
        freeze(self.feature_extractor)
        
        self.encoder.eval()
        self.quantizer.eval()
        self.feature_extractor.eval()
    
    @torch.no_grad()
    def encode_to_z_q(self, y):
        """
        입력 y를 인코딩하여 quantized representation z_q와 discrete token indices를 반환합니다.
        """
        z_e = self.encoder(y)
        z_q, vq_dict = self.quantizer(z_e)
        return z_q, vq_dict['q'].squeeze()
    
    @torch.no_grad()
    def prepare_gru_inputs(self, target_indices, device):
        """
        GRU에 입력할 discrete 토큰 시퀀스 준비 (SOS 토큰 추가 및 마스킹 적용)
        """
        return self.sequential.prepare_inputs(target_indices, device)
    
    def decode_quantized_embeddings(self, firm_features, predicted_indices):
        """
        예측된 토큰 indices로부터 codebook 임베딩을 조회한 후, decoder를 통해 재구성 수행.
        """
        codebook = self.quantizer.get_codebook().to(self.device)
        quantized_embeddings = F.embedding(predicted_indices, codebook)
        y_hat, _ = self.decoder(firm_char=firm_features, inputs=quantized_embeddings)
        return y_hat
    
    def forward(self, firm_char, y, market):
        """
        Forward pass:
          - firm_char: firm-level input
          - y: (B, T, input_channel) 재구성 대상 데이터
          - market: (B, T, market_feature_dim) market 정보
        """
        device = firm_char.device
        
        # Feature 추출 (firm-level)
        firm_features = self.feature_extractor(firm_char)
        market_features = self.market_extractor(market) if self.use_market else None
        
        # VQ Encoder -> Quantizer를 통해 discrete token indices 산출
        z_q, target_indices = self.encode_to_z_q(y)
        input_indices = self.prepare_gru_inputs(target_indices, device)  # (B, T+1)
        
        # GRU 모델을 통해 token 예측 (market 정보가 있다면 함께 사용)
        if self.use_market:
            logits = self.sequential(input_indices[:, :-1], market_features=market_features)
        else:
            logits = self.sequential(input_indices[:, :-1])
        
        # logits로부터 예측 토큰 선택 (argmax)
        predicted_indices = torch.argmax(logits, dim=-1)
        # Decoder: quantized embeddings을 lookup하여 최종 재구성
        y_hat = self.decode_quantized_embeddings(firm_features=firm_features,
                                                 predicted_indices=predicted_indices)
        return logits, target_indices, y_hat