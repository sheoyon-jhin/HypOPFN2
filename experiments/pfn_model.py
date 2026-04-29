"""
PFN-style Time Series Foundation Model.

Differences from our operator-learning model:
  - Each context point (t_i, y_i) is an EXPLICIT token (no patching).
  - Query is also a token (with masked y) appended to the sequence.
  - Self-attention processes (ctx + query) jointly with a structured mask:
      * ctx ↔ ctx     : allowed (full attention among context)
      * ctx → qry     : blocked (so context never sees query y, prevents leakage)
      * qry → ctx     : allowed (queries attend to context to "look up")
      * qry ↔ qry     : blocked off-diagonal (each query independent — no inter-query)
  - Output: point prediction (MSE) or distribution params (NLL Gaussian).
  - Trunk-free: prediction is read directly off the query token at the end.

Inspired by:
  - TabPFN (Hollmann et al., 2023)
  - ForecastPFN (Dooley et al., 2023)
  - TempoPFN (Singh et al., 2024)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sinusoidal time embedding (works for any real-valued t)
# ============================================================
def sinusoidal_t_embed(t, dim, max_period=10000.0):
    """
    t: (..., ) real-valued timestamps.
    dim: embedding dimension (must be even).
    Returns: (..., dim) sinusoidal embedding.
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float().unsqueeze(-1) * freqs  # (..., half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ============================================================
# (t, y) token embedding
# ============================================================
class PairTokenizer(nn.Module):
    """Encode (t, y) into a d_model token. y may be 'masked' for query points."""

    def __init__(self, d_model=512, t_embed_dim=128):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.y_proj = nn.Linear(1, d_model)
        self.t_proj = nn.Linear(t_embed_dim, d_model)
        # Separate flag embedding so model knows "this is a query, ignore y"
        self.is_query_embed = nn.Embedding(2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, t, y, is_query):
        """
        t: (B, N) timestamps
        y: (B, N) values (may be 0 for query tokens — model relies on `is_query` flag)
        is_query: (B, N) bool
        Returns: (B, N, d_model)
        """
        t_emb = sinusoidal_t_embed(t, self.t_embed_dim)
        h = self.y_proj(y.unsqueeze(-1)) + self.t_proj(t_emb)
        h = h + self.is_query_embed(is_query.long())
        return self.norm(h)


# ============================================================
# PFN attention mask helper
# ============================================================
def build_pfn_mask(n_ctx, n_qry, device):
    """
    Build (N, N) attention mask where N = n_ctx + n_qry.

    Convention (PyTorch Transformer): True = blocked.

    Layout:
      [0 .. n_ctx)        : context tokens
      [n_ctx .. N)        : query tokens

    Rules:
      ctx → ctx       allowed (block=False)
      ctx → qry       blocked: context never attends to query y
      qry → ctx       allowed
      qry → other_qry blocked: queries independent (only see themselves + ctx)
    """
    N = n_ctx + n_qry
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    # block ctx → qry
    mask[:n_ctx, n_ctx:] = True
    # block qry → other_qry (off-diagonal in the qry block)
    if n_qry > 0:
        qry_block = torch.ones(n_qry, n_qry, dtype=torch.bool, device=device)
        qry_block.fill_diagonal_(False)
        mask[n_ctx:, n_ctx:] = qry_block
    return mask


# ============================================================
# PFN model
# ============================================================
class PFNTimeSeriesModel(nn.Module):
    """In-context PFN for time series prediction.

    Forward signature:
      y_pred = model(t_ctx, y_ctx, t_qry)
        t_ctx: (B, N_ctx) float
        y_ctx: (B, N_ctx) float (already normalized per sample)
        t_qry: (B, N_qry) float

    Returns:
      If dist_output=False: (B, N_qry) point predictions (MSE-trainable).
      If dist_output=True : (B, N_qry, 2) [mean, log_sigma] for NLL loss.
    """

    def __init__(self, d_model=512, n_layers=8, n_heads=8,
                 t_embed_dim=128, dist_output=False, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.dist_output = dist_output

        self.tokenizer = PairTokenizer(d_model=d_model, t_embed_dim=t_embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.norm_out = nn.LayerNorm(d_model)

        out_dim = 2 if dist_output else 1
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, t_ctx, y_ctx, t_qry, key_padding_mask=None):
        B, Nc = t_ctx.shape
        Nq = t_qry.shape[1]

        # Build tokens
        is_q_ctx = torch.zeros_like(t_ctx, dtype=torch.long)
        is_q_qry = torch.ones_like(t_qry, dtype=torch.long)
        ctx_tok = self.tokenizer(t_ctx, y_ctx, is_q_ctx)              # (B, Nc, D)
        # For query tokens we feed y=0 placeholder; the is_query flag tells model to ignore
        qry_tok = self.tokenizer(t_qry, torch.zeros_like(t_qry), is_q_qry)  # (B, Nq, D)
        all_tok = torch.cat([ctx_tok, qry_tok], dim=1)                # (B, Nc+Nq, D)

        # Mask
        attn_mask = build_pfn_mask(Nc, Nq, t_ctx.device)               # (N, N) bool

        # Transformer expects mask additively (-inf where blocked)
        # PyTorch handles bool mask: True = ignore.
        h = self.transformer(all_tok, mask=attn_mask,
                             src_key_padding_mask=key_padding_mask)   # (B, N, D)
        h = self.norm_out(h)

        # Read off the query positions only
        h_qry = h[:, Nc:, :]                                           # (B, Nq, D)
        out = self.head(h_qry)                                         # (B, Nq, 1 or 2)

        if self.dist_output:
            # Return [mean, log_sigma]; constrain log_sigma to a sane range
            mean = out[..., 0]
            log_sigma = out[..., 1].clamp(-7.0, 3.0)
            return mean, log_sigma
        else:
            return out.squeeze(-1)


# ============================================================
# Loss helpers
# ============================================================
def gaussian_nll(y_true, mean, log_sigma):
    """Gaussian negative log likelihood (mean over all elements)."""
    var = (2 * log_sigma).exp()
    return 0.5 * (((y_true - mean) ** 2) / var + 2 * log_sigma + math.log(2 * math.pi)).mean()


def mse_loss(y_true, y_pred):
    return F.mse_loss(y_pred, y_true)
