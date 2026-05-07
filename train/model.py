import torch

import torch
import torch.nn as nn


class TransformerFusionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # === Self-attention + residual ===
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # === FFN + residual ===
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x, attn_weights


class AttentionFusionModel(nn.Module):
    def __init__(
        self,
        esm_dim,
        fp_dim,
        phys_dim,
        num_classes,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        # === Encoders ===
        def make_encoder(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        self.esm_encoder = make_encoder(esm_dim)
        self.fp_encoder = make_encoder(fp_dim)
        self.phys_encoder = make_encoder(phys_dim)

        # === Learnable modality embeddings ===
        self.modality_embed = nn.Parameter(torch.randn(3, hidden_dim))

        # === CLS token ===
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # === Transformer blocks ===
        self.blocks = nn.ModuleList([
            TransformerFusionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # === Classification head ===
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, esm, fp, phys, return_attention=False):
        # === Encode modalities ===
        e = self.esm_encoder(esm)
        f = self.fp_encoder(fp)
        p = self.phys_encoder(phys)

        # === Stack tokens ===
        tokens = torch.stack([e, f, p], dim=1)

        # === Add modality embeddings ===
        tokens = tokens + self.modality_embed.unsqueeze(0)

        # === Add CLS token ===
        batch_size = tokens.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)

        # === Transformer ===
        attentions = []

        for block in self.blocks:
            x, attn = block(x)
            attentions.append(attn)

        # === CLS output ===
        cls_out = x[:, 0]

        logits = self.classifier(cls_out)

        if return_attention:
            return logits, attentions

        return logits


class AttentionFusionModelv1(torch.nn.Module):
    def __init__(self, esm_dim, fp_dim, phys_dim, num_classes, hidden_dim=128, num_heads=4):
        super().__init__()

        # === encoders (приводим всё к одному размеру) ===
        self.esm_encoder = torch.nn.Sequential(
            torch.nn.Linear(esm_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
        )

        self.fp_encoder = torch.nn.Sequential(
            torch.nn.Linear(fp_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
        )

        self.phys_encoder = torch.nn.Sequential(
            torch.nn.Linear(phys_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
        )

        # === attention ===
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # learnable CLS token
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, hidden_dim))

        # === classifier ===
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, esm, fp, phys):
        # encode features
        e = self.esm_encoder(esm)
        f = self.fp_encoder(fp)
        p = self.phys_encoder(phys)

        # делаем "токены"
        # shape: (batch, 3, hidden_dim)
        # tokens = torch.stack([e, f, p], dim=1)
        tokens = torch.stack([e, f, p], dim=1)

        # добавляем CLS токен
        batch_size = tokens.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        tokens = torch.cat([cls_tokens, tokens], dim=1)
        # shape: (batch, 4, hidden_dim)

        # attention
        attn_out, _ = self.attention(tokens, tokens, tokens)

        # берём CLS токен
        cls_out = attn_out[:, 0]

        return self.classifier(cls_out)
