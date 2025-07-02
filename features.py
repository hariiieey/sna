import numpy as np
import pandas as pd
import networkx as nx
import torch

def extract_node_features(G, historical_graphs=None):
    """Extract comprehensive node features from a NetworkX graph."""
    features = {}
    features['degree_centrality'] = nx.degree_centrality(G)
    features['betweenness_centrality'] = nx.betweenness_centrality(G)
    features['closeness_centrality'] = nx.closeness_centrality(G)
    features['pagerank'] = nx.pagerank(G)
    features['clustering'] = nx.clustering(G.to_undirected())
    features['triangles'] = dict(nx.triangles(G.to_undirected()))
    # Temporal features (optional, stub)
    if historical_graphs:
        features['activity_trend'] = {n: 0.0 for n in G.nodes()}
        features['connection_stability'] = {n: 0.0 for n in G.nodes()}
    df = pd.DataFrame(features).fillna(0)
    return df

def create_positional_encoding(G, dim=64):
    """Create positional encoding for nodes using Laplacian eigenvectors."""
    L = nx.laplacian_matrix(G).toarray()
    eigenvals, eigenvecs = np.linalg.eigh(L)
    if eigenvecs.shape[1] < dim + 1:
        # Pad with zeros if not enough eigenvectors
        pad_width = dim + 1 - eigenvecs.shape[1]
        eigenvecs = np.pad(eigenvecs, ((0,0),(0,pad_width)), 'constant')
    pos_enc = eigenvecs[:, 1:dim+1]  # Skip the first (zero) eigenvalue
    return torch.FloatTensor(pos_enc)

def create_temporal_encoding(timestamps, d_model=128):
    """Create sinusoidal temporal encoding for a list/array of timestamps."""
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    timestamps = np.array(timestamps)
    angle_rads = get_angles(timestamps[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return torch.FloatTensor(angle_rads) 