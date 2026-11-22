import torch
import torch.nn as nn


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.context_size = self.cfg.MODEL.GPT.context_size
        self.n_blocks = self.cfg.MODEL.GPT.n_blocks
        self.vocab_size = self.cfg.TOKENIZER.vocab_size + 4
        self.embed_dim = self.cfg.MODEL.GPT.embed_dim
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.embed_dim)
        self.position_embedding_table = nn.Embedding(self.context_size, self.embed_dim)
        self.decoder_blocks = nn.Sequential(*[DecoderBlock(self.cfg) for _ in range(self.n_blocks)])
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        B, T = x.shape
        eos_positions = torch.argmax((x == 1).int(), dim=1)

        token_embedding = self.token_embedding_table(x)
        positional_embedding = self.position_embedding_table(torch.arange(T, device=x.device))
        x = token_embedding + positional_embedding
        x = self.decoder_blocks(x)
        x = self.layer_norm(x)

        out = x[torch.arange(B, device=x.device), eos_positions]

        return out


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = self.cfg.MODEL.GPT.embed_dim
        self.self_attention = MultiHeadAttention(self.cfg)
        self.feed_forward = FeedForward(self.cfg)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_heads = self.cfg.MODEL.GPT.n_heads
        self.dropout_rate = self.cfg.MODEL.GPT.dropout_rate
        self.embed_dim = self.cfg.MODEL.GPT.embed_dim

        self.heads = nn.ModuleList([AttentionHead(self.cfg) for _ in range(self.n_heads)])
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)

        return out


class AttentionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.context_size = self.cfg.MODEL.GPT.context_size
        self.embed_dim = self.cfg.MODEL.GPT.embed_dim
        self.n_heads = self.cfg.MODEL.GPT.n_heads
        self.head_dim = self.embed_dim // self.n_heads
        self.dropout_rate = self.cfg.MODEL.GPT.dropout_rate

        self.query = nn.Linear(self.embed_dim, self.head_dim)
        self.key = nn.Linear(self.embed_dim, self.head_dim)
        self.value = nn.Linear(self.embed_dim, self.head_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.register_buffer("tril", torch.tril(torch.ones(self.context_size, self.context_size)))

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        out = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        out = out.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        out = nn.functional.softmax(out, dim=-1)
        out = self.dropout(out)
        out = out @ v

        return out


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = self.cfg.MODEL.GPT.embed_dim
        self.dropout_rate = self.cfg.MODEL.GPT.dropout_rate

        self.model = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.ReLU(),
            nn.Linear(4 * self.embed_dim, self.embed_dim),
            nn.Dropout(self.dropout_rate),
        )

    def forward(self, x):
        x = self.model(x)

        return x
