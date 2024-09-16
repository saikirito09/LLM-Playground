import torch
import torch.nn as nn
import math

class NormalizationLayer(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, h, -1).transpose(1, 2), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            dots = dots.masked_fill(mask == 0, float('-inf'))

        attn = self.dropout(dots.softmax(dim=-1))

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = Attention(dim, heads=heads, dropout=dropout)
        self.feed_forward = FeedForward(dim, hidden_dim, dropout=dropout)
        self.norm1 = NormalizationLayer(dim)
        self.norm2 = NormalizationLayer(dim)

    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.feed_forward(self.norm2(x))
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, dim, max_seq_len=5000):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe = torch.zeros(max_seq_len, 1, dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads, hidden_dim, dropout))

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TransformerBlock(dim, heads, hidden_dim, dropout),
                TransformerBlock(dim, heads, hidden_dim, dropout)
            ]))
        self.norm = NormalizationLayer(dim)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for self_attn, cross_attn in self.layers:
            x = self_attn(x, tgt_mask)
            x = cross_attn(x, src_mask)
        return self.norm(x)

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, dim, depth, heads, hidden_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.encoder_embed = nn.Embedding(src_vocab_size, dim)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, dim)
        self.pos_encoder = PositionalEncoder(dim, max_seq_len)

        self.encoder = TransformerEncoder(dim, depth, heads, hidden_dim, dropout)
        self.decoder = TransformerDecoder(dim, depth, heads, hidden_dim, dropout)

        self.to_logits = nn.Linear(dim, tgt_vocab_size)

        self.dim = dim

    def encode(self, src, src_mask):
        src = self.encoder_embed(src) * math.sqrt(self.dim)
        src = self.pos_encoder(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        tgt = self.decoder_embed(tgt) * math.sqrt(self.dim)
        tgt = self.pos_encoder(tgt)
        return self.decoder(tgt, memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.to_logits(dec_output)

def create_transformer(src_vocab_size, tgt_vocab_size, max_seq_len, dim=512, depth=6, heads=8, hidden_dim=2048, dropout=0.1):
    model = TransformerModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dim=dim,
        depth=depth,
        heads=heads,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        dropout=dropout
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
