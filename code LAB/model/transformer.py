import torch
import torch.nn as nn

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, dim=192, depth=6, heads=6, seq_len=256):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(seq_len, dim)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                batch_first=True
            )
            for _ in range(depth)
        ])

        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_embed(x) + self.pos_embed(pos)

        for layer in self.layers:
            h = layer(h)

        return self.fc(self.ln(h))
