import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.graph_attention = TransformerConv(
            in_channels=d_model,
            out_channels=d_model,
            heads=n_heads,
            dropout=dropout,
            concat=False
        )
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, temporal_encoding, batch=None):
        graph_out = self.graph_attention(x, edge_index)
        x = self.norm1(x + self.dropout(graph_out))
        if temporal_encoding is not None:
            temp_out, _ = self.temporal_attention(
                x.unsqueeze(1),
                temporal_encoding.unsqueeze(1),
                temporal_encoding.unsqueeze(1)
            )
            x = self.norm2(x + self.dropout(temp_out.squeeze(1)))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x

class TemporalGraphTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=8, n_layers=6, max_nodes=10000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Embedding(max_nodes, d_model)
        self.temporal_projection = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.link_predictor = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        x = self.input_projection(x)
        pos_enc = self.positional_encoding(torch.arange(x.size(0), device=x.device))
        x = x + pos_enc
        temporal_enc = None
        if hasattr(data, 'temporal_encoding'):
            temporal_enc = self.temporal_projection(data.temporal_encoding)
        for layer in self.layers:
            x = layer(x, edge_index, temporal_enc, batch)
        return x

    def predict_links(self, node_embeddings, candidate_edges):
        src_embeddings = node_embeddings[candidate_edges[:, 0]]
        dst_embeddings = node_embeddings[candidate_edges[:, 1]]
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
        link_probs = self.link_predictor(edge_embeddings)
        return link_probs.squeeze() 