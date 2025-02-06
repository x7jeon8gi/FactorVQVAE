import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from vqtorch.nn import VectorQuant

print('Testing VectorQuant')
# create VQ layer
vq_layer = VectorQuant(
                feature_size=64,     # feature dimension corresponding to the vectors
                num_codes=32,      # number of codebook vectors
                beta=0.98,           # (default: 0.9) commitment trade-off
                kmeans_init=True,    # (default: False) whether to use kmeans++ init
                norm=None,           # (default: None) normalization for the input vectors
                cb_norm=None,        # (default: None) normalization for codebookc vectors
                affine_lr=10.0,      # (default: 0.0) lr scale for affine parameters
                sync_nu=0.2,         # (default: 0.0) codebook synchronization contribution
                replace_freq=20,     # (default: None) frequency to replace dead codes
                dim=-1,              # (default: -1) dimension to be quantized
                ).cuda()

# when `kmeans_init=True` is recommended to warm up the codebook before training
with torch.no_grad():
    z_e = torch.randn(2, 1, 64).cuda()
    vq_layer(z_e)

# standard forward pass
z_e = torch.randn(2, 1, 64).cuda()
z_q, vq_dict = vq_layer(z_e)

print(vq_dict.keys())

print(z_q.shape)
print(vq_dict['z'].shape)
print(vq_dict['loss'])