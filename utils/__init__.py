import os
import pickle
import logging
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from utils.parser import *
from utils.rankloss import RankLoss
from utils.load import freeze, load_pretrained_tok_emb
from utils.seed import seed_everything
from utils.test import run_inference, calc_ic, RankIC
from utils.backtest import calculate_table_metrics
from utils.metric_log import log_metrics_as_bar_chart
def save_model(models_dict: dict, dirname='store', id: str = ''):
    """
    :param models_dict: {'model_name': model, ...}
    """

    if not os.path.isdir(get_root_dir().joinpath(dirname)):
        os.mkdir(get_root_dir().joinpath(dirname))

    id_ = id[:]
    if id != '':
        id_ = '-' + id_
    for model_name, model in models_dict.items():
        torch.save(model.state_dict(), get_root_dir().joinpath(dirname, model_name + id_ + '.ckpt'))