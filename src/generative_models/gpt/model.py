import torch
from torch import nn
from torch.nn import functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(vocab_size, embed_dim)
        self.context_len = context_len

    def forward(self, idx):
        _, time_dim = idx.shape
        assert(time_dim <= self.context_len)
        x_token = self.token_embedding(idx)
        x_position = self.position_embedding(torch.arange(time_dim, device=next(self.parameters()).device))
        x = x_token + x_position
        return x

class CausalSelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, attn_dim, context_len, manual_attention=False):
        super().__init__()
        self.attn_dim = attn_dim
        self.key = nn.Linear(input_dim, attn_dim, bias=False)
        self.query = nn.Linear(input_dim, attn_dim, bias=False)
        self.value = nn.Linear(input_dim, attn_dim, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(context_len, context_len)))
        self._attn_scaling = attn_dim ** -0.5
        self._manual_attention = manual_attention

    def forward(self, x):
        k, q, v = (
                    self.key.forward(x),
                    self.query.forward(x),
                    self.value.forward(x)
        )
        if self._manual_attention:
            batch_dim, time_dim, embed_dim = x.shape
            log_score = k @ q.transpose(-2,-1)
            score = F.softmax(log_score.masked_fill_(self.mask[None, :time_dim,:time_dim] == 0, float("-inf")), dim=-1)
            out = score@v
            return out
        else :
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

class ParallelCausalSelfAttentionLayer(nn.Module):
    def __init__(self, num_parallel, input_dim, attn_dim, context_len, dropout_prob=0.1):
        super().__init__() 
        self.parallel_layers = nn.ModuleList([CausalSelfAttentionLayer(input_dim, attn_dim, context_len) for _ in range(num_parallel)]) 
        n_embed = num_parallel * attn_dim 
        self.linear = nn.Linear(n_embed, n_embed, bias=False) 
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x):
        return self.dropout(self.linear(torch.cat([layer(x) for layer in self.parallel_layers], dim=-1)))

class MLPLayer(nn.Module):
    def __init__(self, n_embed, dropout_prob=0.1):
        super().__init__()
        self.linear_in = nn.Linear(n_embed, 4*n_embed)
        self.linear_out = nn.Linear(4 * n_embed, n_embed)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        return self.dropout(F.relu(self.linear_out(F.relu(self.linear_in(x)))) + x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_attn, context_len, dropout_prob):
        super().__init__()
        assert(n_embed % n_attn == 0)
        num_parallel = n_embed // n_attn
        self.layer_norm_attn = nn.LayerNorm(n_embed)
        self.attn_layer = ParallelCausalSelfAttentionLayer(num_parallel, n_embed, n_attn, context_len, dropout_prob=dropout_prob) 
        self.layer_norm_mlp = nn.LayerNorm(n_embed)
        self.mlp_layer = MLPLayer(n_embed, dropout_prob = dropout_prob)

    def forward(self, x):
        x = self.layer_norm_attn(x)
        x = self.attn_layer(x) + x
        x = self.layer_norm_mlp(x)
        x = self.mlp_layer(x) + x
        return x


class GPTModel(nn.Module):
    def __init__(self, vocab_size, context_len, n_embed, n_attn, num_blocks, dropout_prob=0.1):
        super().__init__()
        self.context_len = context_len
        self.n_embed = n_embed
        self.n_attn = n_attn
        self.num_blocks = num_blocks
        self.embedding = EmbeddingLayer(vocab_size, n_embed, context_len)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embed, n_attn, context_len, dropout_prob=dropout_prob) for _ in range(num_blocks)])
        self.output = nn.Linear(self.n_embed, vocab_size)

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        for block in self.blocks:
            x = block(x)
        x = self.output(x)
        if targets is None:
            return x
        else:
            _, _, channel_dim = x.shape
            loss = F.cross_entropy(x.view(-1, channel_dim), targets.view(-1))
            return x, loss

    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.2)

            if module.bias is not None:
                nn.init_zeros(module.bias)
        if isinstance(module, nn.Ebedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.2)

    def batch_generate(self, idx, n):
        for _ in range(n):
            context_window = idx[:, -self.context_len:]
            logits = self(context_window)
            logits = logits[:, -1, :]
            next_idx = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
