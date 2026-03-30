from sklearn.ensemble import RandomForestRegressor
   
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

import torch
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer


class RandomForest():
    def __init__(self, n_estimators=100, criterion="squared_error"):
        self.model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion)

    def fit(self,x,y):
        self.model.fit(x,y)
    
    def predict(self,x):
        return self.model.predict(x)
    

class SupportVector():
    def __init__(self):
        self.model = MultiOutputRegressor(SVR(kernel="rbf"))

    def fit(self,x,y):
        self.model.fit(x,y)
    
    def predict(self,x):
        return self.model.predict(x)


class KNeighbors():
    def __init__(self, n_neighbors=5, weights="uniform"):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)

    def fit(self,x,y):
        self.model.fit(x,y)
    
    def predict(self,x):
        return self.model.predict(x)
    

class Model500GELU(nn.Module):
    def __init__(self, input_count=2, output_count=2):
        super(Model500GELU, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_count, 200),
            nn.ReLU(),
            nn.Linear(200, 300),
            nn.ReLU(),
            nn.Linear(300, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, output_count)
        )

    def forward(self, x):
        return self.network(x)
    

# ---------------------------------------------------------------------------
# Optimized MLP (V2)
# Improvements over Model500GELU:
#   1. GELU activations  -> smoother gradients
#   2. BatchNorm1d       -> stabilises training, allows higher LR
#   3. Dropout(0.1)      -> regularisation against over-fitting
#   4. Residual skip     -> alleviates vanishing-gradient in deep blocks
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """One residual block: Linear -> BN -> GELU -> Dropout -> Linear -> BN
    with an optional projection shortcut when in/out dims differ."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        # projection shortcut so residual dims always match
        self.shortcut = (
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim))
            if in_dim != out_dim
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.block(x) + self.shortcut(x))


class MultiLayerPerceptronV2(nn.Module):
    """Optimized MLP with residual blocks, BatchNorm, GELU, and Dropout."""
    def __init__(self, input_count: int = 2, output_count: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_count, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            _ResBlock(256, 512, dropout),
            _ResBlock(512, 512, dropout),
            _ResBlock(512, 256, dropout),
            _ResBlock(256, 256, dropout),
        )
        self.head = nn.Linear(256, output_count)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class Transformer(nn.Module):

    def __init__(
        self,
        input_count=2, 
        output_count=2,
        dim_model=200,
        num_heads=2,
        num_encoder_layers=6,
        dim_hidden=200,
        dropout_p=0.1,
    ):
        super(Transformer, self).__init__()

        self.embedding = nn.Linear(input_count, dim_model)

        encoder_layers = TransformerEncoderLayer(dim_model, num_heads, dim_hidden, dropout_p, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layers, num_encoder_layers)

        self.out = nn.Linear(dim_model, output_count)


    def forward(self, src, src_mask=None):

        if src.dim() == 2:
            src = src.unsqueeze(1)
        elif src.dim() != 3:
            raise ValueError("Transformer expects a 2D or 3D tensor input")

        src = self.embedding(src)

        if src_mask is None and src.size(1) > 1:
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)

        transformer_out = self.transformer(src, src_mask)
        out = self.out(transformer_out[:, 0, :])

        return out


class TransformerV2(nn.Module):

    def __init__(
        self,
        input_count=2,
        output_count=2,
        dim_model=192,
        num_heads=6,
        num_encoder_layers=4,
        dim_hidden=384,
        dropout_p=0.1,
    ):
        super(TransformerV2, self).__init__()

        if dim_model % num_heads != 0:
            raise ValueError("dim_model must be divisible by num_heads")

        self.input_count = input_count
        self.feature_token_proj = nn.Linear(1, dim_model)
        self.feature_embedding = nn.Parameter(torch.zeros(1, input_count, dim_model))
        self.context_token = nn.Parameter(torch.zeros(1, 1, dim_model))
        self.token_norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout_p)

        encoder_layer = TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_hidden,
            dropout=dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            norm=nn.LayerNorm(dim_model),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, dim_model),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim_model, output_count),
        )

        nn.init.trunc_normal_(self.context_token, std=0.02)
        nn.init.trunc_normal_(self.feature_embedding, std=0.02)
        nn.init.xavier_uniform_(self.feature_token_proj.weight)
        nn.init.zeros_(self.feature_token_proj.bias)

    def forward(self, src, src_mask=None):
        del src_mask

        if src.dim() != 2:
            raise ValueError("TransformerV2 expects a 2D tensor input")
        if src.size(1) != self.input_count:
            raise ValueError("TransformerV2 input feature size mismatch")

        src = src.unsqueeze(-1)
        feature_tokens = self.feature_token_proj(src)
        feature_tokens = feature_tokens + self.feature_embedding

        batch_size = src.size(0)
        context = self.context_token.expand(batch_size, -1, -1)

        sequence = torch.cat([context, feature_tokens], dim=1)
        sequence = self.dropout(self.token_norm(sequence))
        transformer_out = self.transformer(sequence)

        pooled = transformer_out[:, 0, :]
        return self.head(pooled)