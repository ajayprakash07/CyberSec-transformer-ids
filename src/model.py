import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────
# POSITIONAL ENCODING
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ─────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────

class FlowTransformer(nn.Module):
    def __init__(self,
                 num_features=20,
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 dropout=0.1,
                 num_classes=2):
        super().__init__()
        self.input_embedding = nn.Linear(num_features, d_model)

        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  
            dropout=dropout,
            batch_first=True,             # (batch, seq, features) order
            norm_first=True               
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False   
        )

        self.norm = nn.LayerNorm(d_model)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),                    
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)    # final output - 2 scores
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        # Step 1 — embed raw features into d_model space
        x = self.input_embedding(x)       

        # Step 2 — add positional information
        x = self.pos_encoding(x)           

        # Step 3 — transformer attention
        x = self.transformer(x)            

        # Step 4 — normalize
        x = self.norm(x)                   

        # Step 5 — global average pooling
        x = x.mean(dim=1)                  # (batch, d_model)

        # Step 6 — classify
        out = self.classifier(x)           # (batch, num_classes)

        return out


# ─────────────────────────────────────────────
# MODEL SUMMARY UTILITY
# ─────────────────────────────────────────────

def get_model(device):
    model = FlowTransformer(
        num_features=20,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        num_classes=2
    )

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

    print(f"Model built successfully")
    print(f"Device: {device}")
    print(f"Total trainable parameters: {total_params:,}")

    return model