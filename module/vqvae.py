import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    """
    Simple GRU-based Feature Extractor
    Firm characteristics를 입력받아 hidden state를 반환
    """
    def __init__(self, num_latent, hidden_size, num_layers=1):
        super(FeatureExtractor, self).__init__()
        self.num_latent = num_latent
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.normalize = nn.LayerNorm(num_latent)
        self.linear = nn.Linear(num_latent, num_latent)
        self.leakyrelu = nn.LeakyReLU()
        self.gru = nn.GRU(num_latent, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        #! x: (batch_size, seq_length, num_latent)
        # Apply linear and LeakyReLU activation
        #* layer norm 추가
        x = x.float()
        x = self.normalize(x)
        out = self.linear(x)
        out = self.leakyrelu(out)
        # Forward propagate GRU
        stock_latent, gru_hidden = self.gru(out)
        return stock_latent # (batch_size, seq_length, hidden_size)

class LinearEncoder(nn.Module):
    """
    Linear Encoder for future return y_t
    Linear Layer -> GELU -> Linear Layer
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearEncoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        out = self.linear1(inputs)
        out = self.activation(out)
        out = self.linear2(out)
        return out

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.validate_parameters(hidden_size, num_heads)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.n_head = num_heads

        # Scale factor for the dot-product attention mechanism
        self.scale = nn.Parameter(torch.tensor(self.head_dim ** -0.5), requires_grad=False)
        
        # Initialize layers
        self.init_layers(hidden_size, dropout)

    def validate_parameters(self, hidden_size, num_heads):
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

    def init_layers(self, hidden_size, dropout):
        # Linear layers for computing key, query, and value vectors
        self.W_K = nn.Linear(hidden_size, self.num_heads * self.head_dim)
        self.W_Q = nn.Linear(hidden_size, self.num_heads * self.head_dim)
        self.W_V = nn.Linear(hidden_size, self.num_heads * self.head_dim)

        # Dropout layers
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        
        # Output layer that combines the attention heads
        self.out_layer = nn.Sequential(
            nn.Linear(self.num_heads * self.head_dim, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.shape[0]

        # Compute the query, key, and value vectors from encoder_outputs
        query = self.W_Q(encoder_outputs)
        key = self.W_K(encoder_outputs)
        value = self.W_V(encoder_outputs)

        # Split heads for multi-head attention
        query = self._split_heads(query, batch_size)
        key = self._split_heads(key, batch_size)
        value = self._split_heads(value, batch_size)

        # Apply the attention mechanism
        context, attention_weights = self._apply_attention(query, key, value, batch_size)

        return context, attention_weights  # Return context and attention weights

    def _split_heads(self, x, batch_size):
        # Reshape and transpose for multi-head attention
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _apply_attention(self, query, key, value, batch_size):
        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # Create a mask to block future information
        seq_len = query.size(2)
        mask = torch.triu(torch.ones((seq_len, seq_len), device=query.device), diagonal=1).bool()
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # Compute the context vector
        context = torch.matmul(attention_weights, value)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)
        context = self.out_layer(context)
        
        return context, attention_weights

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = AttentionLayer(hidden_size, num_heads, dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, encoder_outputs):
        """
        encoder_outputs: (B, seq_len, hidden_size)
        """
        context, _ = self.attention(encoder_outputs)
        context = self.dropout1(context)
        context = self.norm1(context + encoder_outputs)

        ff = self.feedforward(context)
        ff = self.dropout2(ff)
        ff = self.norm2(ff + context)

        return ff

class FactorEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, use_attn=True, dropout=0.1, stacks=1):
        super(FactorEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.lin_enc = LinearEncoder(input_size, hidden_size, hidden_size) # * VERY SIMPLE ARCHITECTURE
        # Initialize AttentionLayer
        self.attention = nn.ModuleList([EncoderLayer(hidden_size, num_heads, dropout) for _ in range(stacks)])
        self.use_attn = use_attn
    def forward(self, inputs):
        # Process input through RNNEncoder #todo 이름 바꾸기
        inputs = inputs.float()
        encoder_outputs = self.lin_enc(inputs)

        # Process input through AttentionLayer
        if self.use_attn:
            for layer in self.attention:
                encoder_outputs = layer(encoder_outputs)
        else:
            encoder_outputs = encoder_outputs
        return encoder_outputs


class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.linear(x))
    
class AlphaLayer(LinearLayer):
    def __init__(self, firm_features, hidden_size):
        super(AlphaLayer, self).__init__(firm_features, hidden_size)
        self.mu_layer = nn.Linear(hidden_size, 1)

    def forward(self, firm_char):
        firm_char = super().forward(firm_char)
        alpha_mu = self.mu_layer(firm_char)
        return alpha_mu

class BetaLayer(LinearLayer):
    def __init__(self, firm_features, hidden_size, num_factors):
        super(BetaLayer, self).__init__(firm_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_factors)
    
    def forward(self, firm_char):
        beta = super().forward(firm_char)
        beta_mu = self.linear2(beta)
        return beta_mu

class FactorDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_factors=None):
        super(FactorDecoder, self).__init__()
        # Initialize alpha and beta layers within the class
        self.hidden_size = hidden_size
        self.num_factors = hidden_size if num_factors is None else num_factors
        self.input_size = input_size
        print(f"Decoder :: num_elements: {num_factors}, hidden_size: {hidden_size}")
        self.alpha_layer = AlphaLayer(self.input_size, self.num_factors)
        self.beta_layer = BetaLayer(self.input_size, self.hidden_size, self.num_factors)
        self.factor_layer = nn.Linear(self.hidden_size, self.num_factors)

        self.gru_loading = nn.GRUCell(self.num_factors, self.num_factors)
        self.linear = nn.Linear(self.hidden_size, 1)
        self.activation = nn.GELU()

    def _initialize_hidden_state(self, inputs):
        return torch.zeros(inputs.size(0), self.num_factors, device=inputs.device, dtype=inputs.dtype)
    
    def forward(self, firm_char, inputs, hidden =None):
        alpha = self.alpha_layer(firm_char) # (B, seq_len, 1)
        beta = self.beta_layer(firm_char)   # (B, seq_len, num_factors)

        # inputs: (B, seq_len, num_factors)
        inputs = self.factor_layer(inputs)

        if hidden is None:
            hidden = self._initialize_hidden_state(inputs)

        outputs = []
        for i in range(inputs.size(1)):
            hidden = self.gru_loading(inputs[:, i, :], hidden)
            output = self.activation(hidden)
            # output = self.linear(output)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1) # (B, seq_len, num_factors)

        # (B, seq_len, num_factor) * (B, seq_len, num_factor) -> (B, seq_len, 1)
        y_hat = torch.sum(beta * outputs, dim=-1, keepdim=True) + alpha  # (B, seq_len, 1)

        return y_hat, hidden



#### 예전 ###

# class FactorDecoder(nn.Module):
#     """
#     GRUCell을 활용하는 새로운 FactorDecoder
#     forward 입력 시, latent_factor와 이전 시점의 hidden state를 입력으로 받음
#     """
#     def __init__(self, input_size, hidden_size, output_size=1):
#         super(FactorDecoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.gru_loading = nn.GRUCell(input_size, hidden_size)
#         self.linear = nn.Linear(hidden_size, 1)
#         self.activation = nn.GELU()

#         # self.alpha_layer = AlphaLayer(hidden_size, hidden_size)
#         # self.beta_layer = BetaLayer(hidden_size, hidden_size, num_factors)
#         # self.num_factors = num_factors

#     def forward(self, inputs, hidden =None):
#         """
#         last_hidden: (B, hidden_size)
#         """
#         if hidden is None:
#             hidden = self._initialize_hidden_state(inputs)

#         outputs = []
#         for i in range(inputs.size(1)):
#             hidden = self.gru_loading(inputs[:, i, :], hidden)
#             output = self.activation(hidden)
#             output = self.linear(output)
#             outputs.append(output)

#         """
#         더이상 alpha, beta layer를 사용하지 않는다.
#         """
#         # factor = torch.stack(outputs, dim=1)
        
#         # alpha = self.alpha_layer(last_hidden).unsqueeze(1)  # (B, 1)
#         # beta = self.beta_layer(last_hidden).unsqueeze(1) # (B, num_factors)

#         # value = alpha + (beta * factor).sum(dim=-1, keepdim=True) # (B, seq_len, 1)

#         return torch.stack(outputs, dim=1), hidden
        
    
#     def _initialize_hidden_state(self, inputs):
#         return torch.zeros(inputs.size(0), self.hidden_size, device=inputs.device, dtype=inputs.dtype)
    