import torch
import torch.nn as nn
import torch.nn.functional as F

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
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
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
    """
    Modified FactorEncoder for a sequential VAE.
    
    Returns latent parameters for each time step:
        z_mu: Tensor of shape (B, seq_len, hidden_size)
        z_logvar: Tensor of shape (B, seq_len, hidden_size)
    """
    def __init__(self, input_size, hidden_size, num_heads, use_attn=True, dropout=0.1, stacks=1):
        super(FactorEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Project input into hidden_size dimension.
        self.lin_enc = LinearEncoder(input_size, hidden_size, hidden_size)
        
        # Stacked attention layers for sequential modeling.
        self.attention = nn.ModuleList([EncoderLayer(hidden_size, num_heads, dropout) for _ in range(stacks)])
        self.use_attn = use_attn

        # Fully connected layers to produce per-time-step latent parameters.
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape (B, seq_len, input_size)
        Returns:
            z_mu: Tensor of shape (B, seq_len, hidden_size)
            z_logvar: Tensor of shape (B, seq_len, hidden_size)
        """
        inputs = inputs.float()
        # Project inputs: (B, seq_len, hidden_size)
        encoder_outputs = self.lin_enc(inputs)

        # Apply attention layers if enabled.
        if self.use_attn:
            for layer in self.attention:
                encoder_outputs = layer(encoder_outputs)
        
        # Compute latent parameters for each time step.
        z_mu = self.fc_mu(encoder_outputs)      # (B, seq_len, hidden_size)
        z_logvar = self.fc_logvar(encoder_outputs)  # (B, seq_len, hidden_size)
        return z_mu, z_logvar
    