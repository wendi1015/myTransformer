# myTransformer.py
import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
    def forward(self, x, start=0):
        return x + self.pe[:, start:start + x.size(1), :]

def apply_rope(q, k, seq_pos, dim):
    device = q.device
    theta = 10000.0 ** (torch.arange(0, dim, 2, device=device).float() / dim)
    pos = torch.arange(seq_pos, seq_pos + q.size(2), device=device).float().unsqueeze(-1)
    angles = pos / theta
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    def _rope(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rope = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rope.flatten(-2)
    return _rope(q), _rope(k)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, use_rope=False, rope_max_len=2048):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.rope_max_len = rope_max_len

    def _stable_attn(self, q, k, v, attn_mask=None, training=True):
        ori_dtype = q.dtype
        q = q.float(); k = k.float(); v = v.float()
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        if training:
            attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        return out.to(ori_dtype)

    def forward(self, x, kv=None, attn_mask=None, seq_offset=0):
        kv = x if kv is None else kv
        B, T, C = x.size()
        S = kv.size(1)
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(kv).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(kv).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        if self.use_rope:
            q, k = apply_rope(q, k, seq_offset, self.d_k)
        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)
        ctx = self._stable_attn(q, k, v, attn_mask=attn_mask, training=self.training)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(ctx)

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_rope=False, rope_max_len=2048):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_rope, rope_max_len)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, src_mask=None, pos_start=0):
        h = self.norm1(x)
        h = self.self_attn(h, kv=None, attn_mask=src_mask, seq_offset=pos_start)
        x = x + self.dropout(h)
        h2 = self.norm2(x)
        h2 = self.ffn(h2)
        x = x + self.dropout(h2)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_rope=False, rope_max_len=2048):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_rope, rope_max_len)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_rope=False)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, memory, tgt_mask=None, src_mask=None, pos_start=0):
        h = self.norm1(x)
        h = self.self_attn(h, kv=None, attn_mask=tgt_mask, seq_offset=pos_start)
        x = x + self.dropout(h)
        h2 = self.norm2(x)
        h2 = self.cross_attn(h2, kv=memory, attn_mask=src_mask, seq_offset=0)
        x = x + self.dropout(h2)
        h3 = self.norm3(x)
        h3 = self.ffn(h3)
        x = x + self.dropout(h3)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, num_heads, d_ff, pad_idx, dropout=0.1, use_rope=False, rope_max_len=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = None if use_rope else SinusoidalPositionalEncoding(d_model, max_len=rope_max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, use_rope, rope_max_len) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.pad_idx = pad_idx
        self.use_rope = use_rope
    def forward(self, src_tokens):
        x = self.embed(src_tokens)
        if not self.use_rope:
            x = self.pos(x, start=0)
        src_mask = (src_tokens == self.pad_idx).unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, pos_start=0)
        x = self.norm(x)
        return x, src_mask

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, num_heads, d_ff, pad_idx, dropout=0.1, use_rope=False, rope_max_len=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = None if use_rope else SinusoidalPositionalEncoding(d_model, max_len=rope_max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout, use_rope, rope_max_len) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.pad_idx = pad_idx
        self.use_rope = use_rope
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
    def _build_tgt_mask(self, tgt_tokens):
        pad_mask = (tgt_tokens == self.pad_idx).unsqueeze(1).unsqueeze(2)
        T = tgt_tokens.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_tokens.device), diagonal=1)
        causal = causal.unsqueeze(0).unsqueeze(1)
        return pad_mask | causal
    def forward(self, tgt_in, memory, src_mask):
        x = self.embed(tgt_in)
        if not self.use_rope:
            x = self.pos(x, start=0)
        tgt_mask = self._build_tgt_mask(tgt_in)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, src_mask=src_mask, pos_start=0)
        x = self.norm(x)
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, N=4, num_heads=8, d_ff=1024, pad_idx=0,
                 dropout=0.1, use_rope=True, rope_max_len=2048):
        super().__init__()
        self.encoder_net = Encoder(src_vocab, d_model, N, num_heads, d_ff, pad_idx, dropout, use_rope, rope_max_len)
        self.decoder_net = Decoder(tgt_vocab, d_model, N, num_heads, d_ff, pad_idx, dropout, use_rope, rope_max_len)
    def encoder(self, src_tokens):
        return self.encoder_net(src_tokens)
    def decoder(self, tgt_in, memory, src_mask):
        return self.decoder_net(tgt_in, memory, src_mask)
    def forward(self, src_tokens, tgt_in):
        memory, src_mask = self.encoder(src_tokens)
        return self.decoder(tgt_in, memory, src_mask)
