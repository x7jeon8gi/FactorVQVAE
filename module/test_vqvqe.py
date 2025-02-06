from module.vqvae import FactorEncoder, FactorDecoder, FeatureExtractor
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from vqtorch.nn import VectorQuant


class FactorVQVAE(nn.Module):
    def __init__(
        self, 
        input_channel, 
        hidden_size,
        num_heads, 
        num_features, 
        num_factors, 
        dropout=0.1,
        device=None):

        super(FactorVQVAE, self).__init__()
        self.input_channel = input_channel
        self.hidden_size = hidden_size
        self.num_factors = num_factors
        self.dropout = dropout
        self.device = device
        self.num_heads = num_heads
            
        self.feature_extractor = FeatureExtractor(num_latent = num_features,
                                                  hidden_size = hidden_size)

        self.encoder = FactorEncoder(firm_features = hidden_size, 
                                    input_size = input_channel, 
                                    hidden_size = hidden_size, 
                                    num_heads = num_heads, 
                                    dropout=dropout, 
                                    )

        self.decoder = FactorDecoder(input_size = hidden_size, 
                                    hidden_size= hidden_size,
                                    # num_factors= num_factors, # * factor 개수를 조절하거나 그냥 두거나. 원하는대로.
                                    )
        
        self.vq_layer = VectorQuant(
                    feature_size=hidden_size,   # feature dimension corresponding to the vectors
                    num_codes=num_factors,      # number of codebook vectors
                    beta=0.98,           # (default: 0.9) commitment trade-off
                    kmeans_init=True,    # (default: False) whether to use kmeans++ init
                    norm=None,           # (default: None) normalization for the input vectors
                    cb_norm=None,        # (default: None) normalization for codebookc vectors
                    affine_lr= 2.0,      # (default: 0.0) lr scale for affine parameters
                    sync_nu= 2.0,         # (default: 0.0) codebook synchronization contribution
                    replace_freq=20,     # (default: None) frequency to replace dead codes
                    dim=-1,              # (default: -1) dimension to be quantized
                    ).cuda()
        
    def forward(self, input, firm_char, hidden=None):
        if hidden is None:
            hidden = torch.zeros(input.size(0), self.hidden_size, device=input.device, dtype=input.dtype)
        
        firm_char = self.feature_extractor(firm_char) # (batch_size, hidden_size) 과거 20일의 firm characteristics를 hidden size로 embedding
        z_e  = self.encoder(firm_char, input, hidden=None) # (batch_size, seq_len, hidden_size)

        with torch.no_grad():
            random = torch.rand_like(z_e).cuda()
            self.vq_layer(random)

        z_q, vq_dict = self.vq_layer(z_e) # (batch_size, seq_len, hidden_size)
        print(z_q.shape)
        # z_q = z_q[:,[3,5],:]
        output = self.decoder(firm_char, z_q) # (batch_size, seq_len, input_size)
        return output, vq_dict, z_q
    

# Test code

def test_factor_vq_vae():
    batch_size = 128
    seq_len = 20
    input_channel = 1 # only Close price
    hidden_size = 64
    num_heads = 8
    num_factors = 16 # number of factors
    num_features = 64 # 158 firm characteristics
    
    inputs = torch.randn(batch_size, seq_len, input_channel).to('cuda') # t까지
    firm_char = torch.randn(batch_size, seq_len, num_features).to('cuda') # 실제로는 t-1까지 (하나 lag 되어있음)

    model = FactorVQVAE(input_channel, 
                        hidden_size, 
                        num_heads, 
                        num_features, 
                        num_factors).to('cuda')
    
    output, vq_dict, z_q = model(inputs, firm_char)
    print(vq_dict.keys())
    #print(z_q.shape)
    assert output.shape == (batch_size, seq_len, input_channel)
    assert z_q.shape == (batch_size, seq_len, hidden_size)
    assert len(vq_dict) == 6


    print('Test passed.')

if __name__ == '__main__':
    test_factor_vq_vae()