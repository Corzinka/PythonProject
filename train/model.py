import torch

class ConcatModel(torch.nn.Module):
    def __init__(self, num_classes, hidden_dim=128):
        super().__init__()

        # === classifier ===
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.LayerNorm(80),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(128, 16),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(64, num_classes)
        )

    def forward(self, esm, fp, phys):
        cat = torch.cat([esm, fp, phys], dim=1)

        return self.classifier(cat)



class AttentionFusionModel(torch.nn.Module):
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
