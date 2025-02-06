import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
from module.vqvae import FactorEncoder, FactorDecoder, FeatureExtractor
from vqtorch.nn import VectorQuant
import os
import sys
#from x_transformers import Encoder, ContinuousTransformerWrapper
import math
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
from utils import freeze, get_root_dir, load_pretrained_tok_emb


class Transformer(nn.Module):
    def __init__(self,
                 num_tokens: int,
                 codebook_sizes: int,
                 hidden_size: int,
                 embed_dim: int,
                 n_layers: int,
                 heads: int,
                 ff_mult: int,
                 use_rmsnorm: bool,
                 use_pretrained_tok_emb: bool ,
                 freeze_pretrained_tokens: bool ,
                 pretrained_tok_emb: nn.Parameter = None,
                 **kwargs):
        
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_sizes = codebook_sizes
        self.embed_dim = embed_dim

        self.tok_emb = nn.Embedding(self.codebook_sizes+1, self.embed_dim) # +1 for mask token
        if use_pretrained_tok_emb:
            assert self.embed_dim == pretrained_tok_emb.shape[1], "embed_dim of the transformer should be the same as pretrained_tok_emb"
            load_pretrained_tok_emb(pretrained_tok_emb, self.tok_emb, freeze_pretrained_tokens)

        self.pos_emb = nn.Embedding(self.num_tokens+1, self.embed_dim)
        self.blocks = ContinuousTransformerWrapper(
                dim_in = self.embed_dim,
                dim_out= self.embed_dim,
                max_seq_len= self.num_tokens +1,
                attn_layers= Encoder(
                    dim= hidden_size,
                    depth= n_layers,
                    heads= heads,
                    ff_mult= ff_mult,
                    use_rmsnorm= use_rmsnorm,
                    use_abs_pos_emb= False)
                )
            
        self.Token_Prediction = nn.Sequential(*[
            nn.Linear(in_features=embed_dim, out_features=embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim, eps=1e-12)
        ])

        self.bias = nn.Parameter(torch.zeros(self.num_tokens, self.codebook_sizes+1))
        self.ln = nn.LayerNorm(self.embed_dim, eps=1e-12)
        self.drop = nn.Dropout(p=0.01)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, embed):
        device = embed.device

        token_embeddings = self.tok_emb(embed)
        n = token_embeddings.shape[1]
        postion_embeddings = self.pos_emb.weight[:n,: ]
        embed = self.drop(self.ln(token_embeddings + postion_embeddings))
        embed = self.blocks(embed)
        embed = self.Token_Prediction(embed)

        logits = torch.matmul(embed, self.tok_emb.weight.t()) + self.bias # TODO 보다 깔끔히 변경
        logits = logits[:, :, :-1] # remove mask token embedding
        return logits


class BidirectionalTransformer(nn.Module):

    def __init__(self,
                 temperature: float,
                 T: int,
                 config: dict,
                 **kwargs):
        super().__init__()
        self.temperature = temperature
        self.T = T
        self.config = config
        
        self.run_name       = config['train']['run_name']
        self.mask_token_ids = config['vqvae']['num_factors'] # same as codebook size
        self.gamma          = self.gamma_func("cosine") # Masking function
        self.num_factors    = config['vqvae']['num_factors']
        self.device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # define models
        self.dim    = config['vqvae']['hidden_size']
        self.in_channels = config['vqvae']['input_channel']
        self.dropout       = config['vqvae']['dropout'] # 0.1
        self.num_heads     = config['vqvae']['num_heads']
        self.num_features  = config['vqvae']['num_features']

        self.feature_extractor = FeatureExtractor(num_latent = self.num_features,
                                                 hidden_size = self.dim)

        self.encoder = FactorEncoder(input_size = self.in_channels, 
                                     hidden_size = self.dim, 
                                     num_heads = self.num_heads,
                                     use_attn= True,
                                     dropout = self.dropout,
                                    )

        self.decoder = FactorDecoder(input_size = self.dim, hidden_size = self.dim,
                                     num_factors= config['vqvae']['num_elements'])
        
        # self.quantizer = VectorQuant(
        #             feature_size = self.dim,                                # feature dimension corresponding to the vectors
        #             num_codes=self.num_factors,                             # number of codebook vectors
        #             beta=self.config['quantizer']['beta'],                  # (default: 0.9) commitment trade-off
        #             kmeans_init=self.config['quantizer']['kmeans_init'],    # (default: False) whether to use kmeans++ init
        #             norm=None,                                              # (default: None) normalization for the input vectors
        #             cb_norm=None,                                           # (default: None) normalization for codebookc vectors
        #             affine_lr= self.config['quantizer']['affine_lr'],       # (default: 0.0) lr scale for affine parameters
        #             sync_nu=self.config['quantizer']['sync_nu'],            # (default: 0.0) codebook synchronization contribution
        #             replace_freq= self.config['quantizer']['replace_freq'], # (default: None) frequency to replace dead codes
        #             dim=-1,                                                 # (default: -1) dimension to be quantized
        #             )

        self.quantizer = VectorQuantize(
            dim = self.dim,
            codebook_dim= self.num_factors,
            decay = 0.9,
            commitment_weight= 0.25
        ).cuda()
        
        # load trained models for encoder, decoder, and quantizer
        self.load_pretrained_model(config)

        self.num_tokens = self.config['transformer']['num_tokens'] # ?: Max Seq length
        embed = nn.Parameter(copy.deepcopy(self.quantizer.get_codebook()))

        #* follow vqvae setting
        self.transformer = Transformer(
            num_tokens          = self.num_tokens,         
            codebook_sizes      = self.config['vqvae']['num_factors'], # self.config['transformer']['codebook_sizes'],
            hidden_size         = self.config['transformer']['hidden_size'],
            embed_dim           = self.config['transformer']['embed_dim'],  # self.config['vqvae']['hidden_size'], # self.config['transformer']['embed_dim'],
            n_layers            = self.config['transformer']['n_layers'],
            heads               = self.config['transformer']['heads'],
            ff_mult             = self.config['transformer']['ff_mult'],
            use_rmsnorm         = self.config['transformer']['use_rmsnorm'],
            use_pretrained_tok_emb   = self.config['transformer']['use_pretrained_tok_emb'],
            freeze_pretrained_tokens = self.config['transformer']['freeze_pretrained_token'],
            pretrained_tok_emb = embed,
        )

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError

    def load_pretrained_model(self, config):
        
        saved_model = config['transformer']['saved_model']
        if saved_model == False:
            print("No pretrained model is loaded. Please provide the path to the pretrained model.")
            return
        # check end with .ckpt
        if saved_model and not saved_model.endswith('.ckpt'):
            saved_model += '.ckpt'

        checkpoint_path = Path(get_root_dir()).joinpath('checkpoints', saved_model)
        checkpoint = torch.load(checkpoint_path)['state_dict']

        # feature_extractor 모듈에 대한 가중치를 불러옵니다.
        feature_extractor_state_dict = {k.replace('feature_extractor.', ''): v for k, v in checkpoint.items() if k.startswith('feature_extractor')}
        self.feature_extractor.load_state_dict(feature_extractor_state_dict)

        # encoder 모듈에 대한 가중치를 불러옵니다.
        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in checkpoint.items() if k.startswith('encoder')}
        self.encoder.load_state_dict(encoder_state_dict)

        # decoder 모듈에 대한 가중치를 불러옵니다.
        decoder_state_dict = {k.replace('decoder.', ''): v for k, v in checkpoint.items() if k.startswith('decoder')}
        self.decoder.load_state_dict(decoder_state_dict)

        # quantizer 모듈에 대한 가중치를 불러옵니다.
        quantizer_state_dict = {k.replace('quantizer.', ''): v for k, v in checkpoint.items() if k.startswith('quantizer')}
        self.quantizer.load_state_dict(quantizer_state_dict)

        freeze(self.encoder)
        freeze(self.decoder)
        freeze(self.quantizer)
        freeze(self.feature_extractor)

        self.encoder.eval()
        self.decoder.eval()
        self.quantizer.eval()
        self.feature_extractor.eval()

    def load(self, model, dirname, fname):
        path = Path(dirname) / fname
        try: 
            model.load_state_dict(torch.load(path))
            print(f"Loaded {fname}")
        except FileNotFoundError:
            print(f"Failed to load {fname}")
    
    @torch.no_grad()
    def encode_to_z_q(self, y):

        z_e = self.encoder(y)
        # z_q, vq_dict = self.quantizer(z_e) # (batch_size, seq_len, hidden_size)
        # return z_q, vq_dict['q'].squeeze() #?(z, z_q, d, q, loss, perplexity)

        z_q, indices, commitment_loss = self.quantizer(z_e)

        return z_q, commitment_loss, indices

    def forward(self, firm_char, y):
        
        device = firm_char.device
        firm_char = self.feature_extractor(firm_char)

        z_q, idx = self.encode_to_z_q(y)

        # randomly sample `t`
        t = np.random.uniform(0, 1)

        # create masks: top-k (MaskGIT)
        n_masks = math.floor(self.gamma(t) * idx.shape[1])
        rand = torch.rand(idx.shape, device=device)  # (b n)
        mask = torch.zeros(idx.shape, dtype=torch.bool, device=device)
        mask.scatter_(dim=1, index=rand.topk(n_masks, dim=1).indices, value=True)

        # masked tokens
        masked_indices = self.mask_token_ids * torch.ones_like(idx, device=device)  # (b n)
        idx_M = mask * idx + (~mask) * masked_indices  # (b n); `~` reverses bool-typed data

        # prediction
        logit = self.transformer(idx_M.detach())
        target = idx.detach()

        return logit, target

    def create_input_tokens_normal(self, num, num_tokens, mask_token_ids, device):
        """
        return masked tokens
        """
        
        blank_tokens = torch.ones((num, num_tokens), device = device)
        masked_tokens = mask_token_ids * blank_tokens
        return masked_tokens.to(torch.int32)

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0, device='cpu'):
        """
        mask_len: (b 1)
        probs: (b n); also for the confidence scores

        This version keeps `mask_len` exactly.
        """
        def log(t, eps=1e-20):
            return torch.log(t.clamp(min=eps))

        def gumbel_noise(t):
            """
            Gumbel max trick: https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch
            """
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))

        confidence = torch.log(probs + 1e-5) + temperature * gumbel_noise(probs).to(device)  # Gumbel max trick; 1e-5 for numerical stability; (b n)
        mask_len_unique = int(mask_len.unique().item())
        masking_ind = torch.topk(confidence, k=mask_len_unique, dim=-1, largest=False).indices  # (b k)
        masking = torch.zeros_like(confidence).to(device)  # (b n)
        for i in range(masking_ind.shape[0]):
            masking[i, masking_ind[i].long()] = 1.
        masking = masking.bool()
        return masking

    @torch.no_grad()
    def iterative_decoding(self, num=1, mode="cosine", guidance_scale: float =1.):
        """
        iterative decoding & sampling token indices
        num: number of samples
        return: sampled token indices
        """ 
        # create all masked tokens
        s = self.create_input_tokens_normal(num, self.num_tokens, self.mask_token_ids, self.device)
        
        unknown_number_in_the_beginning = torch.sum(s == self.mask_token_ids, dim=1) # (batch, )
        
        gamma = self.gamma_func(mode)

        for t in range(self.T):

            logit = self.transformer(s)
            sample_ids = torch.distributions.categorical.Categorical(logits=logit).sample() # (batch, n)
            unknown_map = (s == self.mask_token_ids) # (batch, n)
            sample_ids = torch.where(unknown_map , sample_ids, s)

            # create masking according to `t`
            ratio = 1. * (t + 1) / self.T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)

            probs = F.softmax(logit, dim=-1)
            selected_prob = torch.gather(probs, dim=-1, index=sample_ids.unsqueeze(-1)).squeeze(-1)
            selected_prob = torch.where(unknown_map, selected_prob, torch.Tensor([torch.inf]).to(self.device))

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)
            mask_len = torch.clip(mask_len, min=0.)

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_prob, temperature=self.temperature * (1. - ratio), device=self.device)

            # Masks tokens with lower confidence.
            s= torch.where(masking, self.mask_token_ids, sample_ids)  # (b n)

        return s

    @torch.no_grad()
    def decode_token_idx_to_timeseries(self, firm_char:torch.Tensor,  s:torch.Tensor):
        """
        It takes token embedding indices and decodes them to time series.
        :param s: token embedding index
        """
        
        quantize = F.embedding(s, self.quantizer.get_codebook())
        firm_char = self.feature_extractor(firm_char)
        y_hat = self.decoder(firm_char, quantize)

        return y_hat