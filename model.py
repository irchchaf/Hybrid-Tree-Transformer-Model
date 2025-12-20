import torch
import torch.nn as nn


class APEncoder(nn.Module):
    """
    Encodes Access Point (AP) positions into a fixed-dimensional embedding.

    Args:
        hidden_dim (int): Dimension of the hidden embedding.

    Input:
        ap_positions (Tensor): Shape (batch_size, num_APs, 2), AP coordinates.

    Output:
        Tensor: Shape (batch_size, hidden_dim), aggregated AP embedding.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, ap_positions: torch.Tensor) -> torch.Tensor:
        ap_features = self.mlp(ap_positions)
        return ap_features.mean(dim=1)


class BinaryTreeCompressor(nn.Module):
    """
    Compresses per-user descriptors into a single root embedding using
    a binary tree merge strategy.

    Args:
        input_dim (int): Dimension of the fused user+AP descriptor.
        d_model (int): Dimension of the compressed representation.

    Input:
        x (Tensor): Shape (batch_size, num_users, input_dim).

    Output:
        Tensor: Shape (batch_size, d_model), compressed root embedding.
    """
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.merge = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        while x.size(1) > 1:
            if x.size(1) % 2 == 1:
                pad = torch.zeros(x.size(0), 1, x.size(2), device=x.device)
                x = torch.cat([x, pad], dim=1)
            x = x.view(x.size(0), x.size(1) // 2, 2 * x.size(2))
            x = self.merge(x)
        return x.squeeze(1)


class RootTransformer(nn.Module):
    """
    Transformer encoder applied to the root embedding.

    Args:
        d_model (int): Dimension of the root embedding.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer encoder layers.
        dropout (float): Dropout rate.

    Input:
        root (Tensor): Shape (batch_size, d_model).

    Output:
        Tensor: Shape (batch_size, d_model), enriched root embedding.
    """
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, root: torch.Tensor) -> torch.Tensor:
        x = root.unsqueeze(0)  # (1, batch_size, d_model)
        x = self.encoder(x)
        return x.squeeze(0)


class SharedPerUserDecoder(nn.Module):
    """
    Decodes per-user uplink and downlink powers from user descriptors
    and the enriched root embedding.

    Args:
        user_descriptor_dim (int): Dimension of per-user descriptor.
        root_dim (int): Dimension of root embedding.

    Input:
        user_descriptors (Tensor): Shape (batch_size, num_users, user_descriptor_dim).
        root (Tensor): Shape (batch_size, root_dim).

    Output:
        Tensor: Shape (batch_size, num_users, 2), UL and DL powers.
    """
    def __init__(self, user_descriptor_dim: int, root_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(user_descriptor_dim + root_dim, root_dim),
            nn.ReLU(),
            nn.Linear(root_dim, 2),
            nn.Sigmoid()
        )

    def forward(self, user_descriptors: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
        root_expanded = root.unsqueeze(1).expand(-1, user_descriptors.size(1), -1)
        x = torch.cat([user_descriptors, root_expanded], dim=-1)
        return self.fc(x)


class HybridPowerAllocator(nn.Module):
    """
    Full hybrid model that combines AP encoding, user embedding,
    binary tree compression, transformer enrichment, and per-user decoding.

    Args:
        ap_embed_dim (int): Dimension of AP and user embeddings.
        d_model (int): Dimension of compressed root embedding.
        num_heads (int): Number of transformer attention heads.
        num_layers (int): Number of transformer encoder layers.

    Input:
        user_positions (Tensor): Shape (batch_size, num_users, 2).
        ap_positions (Tensor): Shape (batch_size, num_APs, 2).

    Output:
        Tensor: Shape (batch_size, num_users, 2), UL and DL powers.
    """
    def __init__(self, ap_embed_dim: int, d_model: int, num_heads: int, num_layers: int):
        super().__init__()
        self.ap_encoder = APEncoder(ap_embed_dim)
        self.user_embed = nn.Linear(2, ap_embed_dim)
        self.compressor = BinaryTreeCompressor(ap_embed_dim * 2, d_model)
        self.transformer = RootTransformer(d_model, num_heads, num_layers)
        self.decoder = SharedPerUserDecoder(ap_embed_dim * 2, d_model)

    def forward(self, user_positions: torch.Tensor, ap_positions: torch.Tensor) -> torch.Tensor:
        ap_embedding = self.ap_encoder(ap_positions)
        user_embedding = self.user_embed(user_positions)
        ap_expanded = ap_embedding.unsqueeze(1).expand(-1, user_positions.size(1), -1)
        fused_input = torch.cat([user_embedding, ap_expanded], dim=-1)
        root = self.compressor(fused_input)
        enriched_root = self.transformer(root)
        powers = self.decoder(fused_input, enriched_root)
        ul = powers[:, :, 0]
        dl = powers[:, :, 1]
        return torch.cat([ul, dl], dim=-1)
