# FactorVQVAE: Discrete latent factor model via Vector Quantized Variational Autoencoder

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="images/stage_overview.png" alt="Stage Overview" width="400"/>
        <br>
        <em>Figure 1: Two-stage training pipeline</em>
      </td>
      <td align="center">
        <img src="images/ls-CSI.png" alt="CSI Performance Results" width="400"/>
        <br>
        <em>Figure 2: Performance results on CSI data</em>
      </td>
    </tr>
  </table>
</div>

## Overview

Official PyTorch implementation of FactorVQVAE for learning discrete latent factors from financial data

## Key Features

- **Two-stage training pipeline**: VQ-VAE training followed by GPT-based generative modeling
- **Alpha158 features**: Comprehensive stock analysis using Qlib's Alpha158 characteristics
- **Vector Quantization**: Compression of continuous features into discrete codebook representations

## Model Architecture

### Stage 1: VQ-VAE Training
- **Feature Extractor**: GRU-based time series feature extraction
- **Vector Quantizer**: Continuous-to-discrete codebook mapping
- **Attention Layer**: Multi-head attention mechanism
- **Linear Encoder/Decoder**: Return encoding and decoding

### Stage 2: GPT-based Generative Model
- **Transformer Architecture**: Autoregressive sequence generation
- **Market Feature Integration**: Market-wide feature incorporation
- **Rank Loss**: Enhanced return ranking prediction

## Project Structure

```
FactorVQVAE-General/
├── configs/                 # Model configuration files
├── data/                    # Dataset classes and Chinese stock data
├── module/                  # Core model implementations
├── trainer/                 # Training modules
├── utils/                   # Utility functions
├── vqtorch/                 # Vector Quantization library
├── stage1.py               # Stage 1 training script
├── stage2_gpt.py          # Stage 2 training script
└── stage2_gpt_sweep.py    # Hyperparameter sweep script
```

## Installation

### Requirements
```bash
pip install torch pytorch-lightning
pip install qlib
pip install pandas numpy
pip install tensorboard
pip install pyyaml
pip install cupy-cuda11x # or pip install cupy-cuda12x
```

### VQTorch Installation
For the vector quantization library, please refer to the official repository at [@https://github.com/minyoungg/vqtorch](https://github.com/minyoungg/vqtorch) for detailed installation instructions.

**Important**: Make sure to install the correct CuPy version that is compatible with your CUDA version before installing vqtorch:

```bash
# For CUDA 12.x versions
pip install cupy-cuda12x

# For CUDA 11.x versions  
pip install cupy-cuda11x

# Then install vqtorch
git clone https://github.com/minyoungg/vqtorch
cd vqtorch
pip install -e .
```

### Qlib Data Setup

Due to Qlib's data accessibility considerations, we recommend using opensource data sources. Please refer to the official Qlib repository at [@https://github.com/microsoft/qlib](https://github.com/microsoft/qlib) for detailed data setup instructions.

#### Chinese Stock Data
For Chinese stock market data, you can use the following commands:

```bash
# Download pre-processed Chinese stock data (recommended)
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
rm -f qlib_bin.tar.gz
```

Alternatively, you can use the official Qlib data downloader:
```bash
# Download Qlib Chinese stock data (official method)
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

#### US Stock Data
For US stock market data, we use Qlib's Yahoo data collector from [@https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo](https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo).

Market index data can be obtained using:
```bash
# Parse instruments for market indices
python collector.py --index_name SP500 --qlib_dir ~/.qlib/qlib_data/us_data --method parse_instruments

# Parse new companies
python collector.py --index_name SP500 --qlib_dir ~/.qlib/qlib_data/us_data --method save_new_companies

# Supported index_name: SP500, NASDAQ100, DJIA, SP400
# For more options
python collector.py --help
```

## Usage

### Stage 1: VQ-VAE Training
```bash
python stage1.py
```

### Stage 2: GPT Model Training
```bash
python stage2_gpt.py
```

### Hyperparameter Tuning
```bash
python stage2_gpt_sweep.py
```

## Configuration

The model configuration is managed through `configs/config.yaml`, which includes:

- **VQ-VAE settings**: Feature dimensions, codebook size, hidden layers
- **Transformer settings**: Architecture parameters, attention heads, layers
- **Training settings**: Batch size, learning rate, epochs, device configuration
- **Data settings**: Stock universe, time periods, window size


## Citation

If you find this work useful for your research, we would be grateful if you could cite our paper:

```bibtex
@article{kim2025factorvqvae,
  title={FactorVQVAE: Discrete latent factor model via Vector Quantized Variational Autoencoder},
  author={Kim, Namhyoung and Ock, Seung Eun and Song, Jae Wook},
  journal={Knowledge-Based Systems},
  volume={318},
  pages={113460},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgments

We are grateful to the authors and contributors of the following projects that made this work possible:

- **VQTorch** ([@https://github.com/minyoungg/vqtorch](https://github.com/minyoungg/vqtorch)): For providing an efficient and well-implemented vector quantization library
- **MinGPT** ([@https://github.com/karpathy/minGPT](https://github.com/karpathy/minGPT)): For the clean and educational GPT implementation that inspired our transformer architecture
- **Qlib** ([@https://github.com/microsoft/qlib](https://github.com/microsoft/qlib)): For the comprehensive quantitative investment platform and Alpha158 features

We deeply appreciate the open-source community and the researchers who have contributed to these foundational works.
