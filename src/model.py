import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────
# POSITIONAL ENCODING
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Injects position information into each flow vector.

    The Transformer attention mechanism is order-blind —
    it sees all flows simultaneously with no sense of which
    came first. Positional encoding fixes this by adding
    a unique pattern to each position.

    Position 0 gets a different pattern than position 1, 2, 3...
    so the model can distinguish flow order.

    Uses sine and cosine functions at different frequencies —
    this is the original formula from "Attention is All You Need".
    """

    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe = positional encoding matrix
        # shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # position indices: 0, 1, 2, ..., max_len-1
        # unsqueeze adds a dimension: shape becomes (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # scaling term — creates different frequencies for each dimension
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # even dimensions → sine wave
        pe[:, 0::2] = torch.sin(position * div_term)

        # odd dimensions → cosine wave
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dimension: (1, max_len, d_model)
        # the 1 allows it to broadcast across any batch size
        pe = pe.unsqueeze(0)

        # register_buffer = saved with model but NOT trained
        # (positional encoding is fixed math, not learned)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # add positional encoding to each flow vector
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────

class FlowTransformer(nn.Module):
    """
    Transformer model for network flow sequence classification.

    Args:
        num_features : number of input features per flow (20)
        d_model      : hidden dimension size (64)
        nhead        : number of attention heads (must divide d_model)
        num_layers   : number of stacked transformer encoder layers
        dropout      : dropout rate for regularization
        num_classes  : 2 for binary (BENIGN / ATTACK)
    """

    def __init__(self,
                 num_features=20,
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 dropout=0.1,
                 num_classes=2):
        super().__init__()

        # ── Step 1: Input Embedding ──────────────────────────
        # Projects 20 raw features → d_model dimensions
        # This gives the model a richer space to work in
        self.input_embedding = nn.Linear(num_features, d_model)

        # ── Step 2: Positional Encoding ──────────────────────
        # Injects flow order information
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout
        )

        # ── Step 3: Transformer Encoder ──────────────────────
        # One encoder layer = one round of attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # internal FF layer is 4x d_model
            dropout=dropout,
            batch_first=True,             # (batch, seq, features) order
            norm_first=True               # pre-norm: more stable training
        )

        # Stack num_layers encoder layers on top of each other
        # Each layer refines the attention from the previous layer
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False    # avoids a PyTorch warning
        )

        # ── Step 4: Layer Normalization ───────────────────────
        # Stabilizes values after transformer
        self.norm = nn.LayerNorm(d_model)

        # ── Step 5: Classifier ────────────────────────────────
        # Global avg pooling happens in forward()
        # Then this MLP makes the final decision
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),                    # adds non-linearity
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)    # final output: 2 scores
        )

        # ── Weight Initialization ─────────────────────────────
        # Better starting weights = faster, more stable training
        self._init_weights()

    def _init_weights(self):
        """
        Initialize linear layer weights using Xavier uniform.
        Xavier keeps gradient magnitudes stable at the start of training.
        Without this, gradients can vanish or explode early on.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass — how data flows through the model.

        x shape at each step:
          input:               (batch, seq_len, num_features)
          after embedding:     (batch, seq_len, d_model)
          after pos encoding:  (batch, seq_len, d_model)
          after transformer:   (batch, seq_len, d_model)
          after avg pooling:   (batch, d_model)
          after classifier:    (batch, num_classes)
        """

        # Step 1 — embed raw features into d_model space
        x = self.input_embedding(x)        # (batch, seq_len, d_model)

        # Step 2 — add positional information
        x = self.pos_encoding(x)           # (batch, seq_len, d_model)

        # Step 3 — transformer attention
        x = self.transformer(x)            # (batch, seq_len, d_model)

        # Step 4 — normalize
        x = self.norm(x)                   # (batch, seq_len, d_model)

        # Step 5 — global average pooling
        # collapse the sequence dimension by averaging all flow vectors
        # each flow's contribution is averaged into one summary vector
        x = x.mean(dim=1)                  # (batch, d_model)

        # Step 6 — classify
        out = self.classifier(x)           # (batch, num_classes)

        return out


# ─────────────────────────────────────────────
# MODEL SUMMARY UTILITY
# ─────────────────────────────────────────────

def get_model(device):
    """
    Builds model, moves to device, prints summary.
    Returns model ready for training.
    """
    model = FlowTransformer(
        num_features=20,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        num_classes=2
    )

    model = model.to(device)

    # count trainable parameters
    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

    print(f"Model built successfully")
    print(f"Device: {device}")
    print(f"Total trainable parameters: {total_params:,}")

    return model