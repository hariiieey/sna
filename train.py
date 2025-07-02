import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import numpy as np
from data_utils import load_temporal_data, create_temporal_snapshots
from features import extract_node_features, create_positional_encoding, create_temporal_encoding
from model import TemporalGraphTransformer
from dataset import TemporalGraphDataset
from evaluate import evaluate_model
import pandas as pd
import csv

class LinkPredictionDataset(Dataset):
    def __init__(self, samples, node_features_dict, pos_enc_dict, temporal_enc_dict):
        self.samples = samples
        self.node_features_dict = node_features_dict
        self.pos_enc_dict = pos_enc_dict
        self.temporal_enc_dict = temporal_enc_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        last_graph = sample['historical_graphs'][-1]
        node_features = self.node_features_dict[last_graph['timestamp']]
        pos_enc = self.pos_enc_dict[last_graph['timestamp']]
        temporal_enc = self.temporal_enc_dict[last_graph['timestamp']]
        # Map node IDs to contiguous indices
        node_list = list(last_graph['graph'].nodes())
        node_id_map = {nid: i for i, nid in enumerate(node_list)}
        # Remap node features
        node_features = node_features.loc[node_list]
        # Remap edge indices
        import torch_geometric
        import torch
        edges = np.array(list(last_graph['graph'].edges()))
        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor([[node_id_map[src], node_id_map[dst]] for src, dst in edges], dtype=torch.long).T
        x = torch.tensor(node_features.values, dtype=torch.float)
        # Remap positive and negative edge indices
        pos_edges = sample['positive_edges']
        neg_edges = sample['negative_edges']
        if len(pos_edges) == 0:
            pos_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            pos_edge_index = torch.tensor([[node_id_map[src], node_id_map[dst]] for src, dst in pos_edges if src in node_id_map and dst in node_id_map], dtype=torch.long).T
        if len(neg_edges) == 0:
            neg_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            neg_edge_index = torch.tensor([[node_id_map[src], node_id_map[dst]] for src, dst in neg_edges if src in node_id_map and dst in node_id_map], dtype=torch.long).T
        data = torch_geometric.data.Data(
            x=x,
            edge_index=edge_index,
            pos_edge_index=pos_edge_index,
            neg_edge_index=neg_edge_index,
            temporal_encoding=temporal_enc
        )
        return data

def main():
    # 1. Load data
    df = load_temporal_data('CollegeMsg.txt')
    # 2. Create temporal snapshots
    snapshots = create_temporal_snapshots(df, window_size='1D')
    # 3. Feature extraction
    node_features_dict = {}
    pos_enc_dict = {}
    temporal_enc_dict = {}
    for snap in snapshots:
        G = snap['graph']
        node_features = extract_node_features(G)
        pos_enc = create_positional_encoding(G, dim=64)
        # Use the same timestamp for all nodes in this snapshot
        temporal_enc = create_temporal_encoding(
            np.array([int(snap['timestamp'].timestamp())]*len(G.nodes())), d_model=64
        )
        node_features_dict[snap['timestamp']] = node_features
        pos_enc_dict[snap['timestamp']] = pos_enc
        temporal_enc_dict[snap['timestamp']] = temporal_enc
    # 4. Create dataset
    tgd = TemporalGraphDataset(snapshots, prediction_horizon=1)
    samples = tgd.create_training_samples()
    # 5. Split train/val
    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]
    train_dataset = LinkPredictionDataset(train_samples, node_features_dict, pos_enc_dict, temporal_enc_dict)
    val_dataset = LinkPredictionDataset(val_samples, node_features_dict, pos_enc_dict, temporal_enc_dict)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # 6. Model
    input_dim = train_dataset[0].x.shape[1]
    model = TemporalGraphTransformer(input_dim=input_dim, d_model=64, n_heads=4, n_layers=2, dropout=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCELoss()
    best_val_auc = 0
    epoch_results = []
    for epoch in range(10):  # For demo, use 10 epochs
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            # Skip batches with no positive or negative edges
            if (batch.pos_edge_index.dim() < 2 or batch.neg_edge_index.dim() < 2 or
                batch.pos_edge_index.size(1) == 0 or batch.neg_edge_index.size(1) == 0):
                continue
            optimizer.zero_grad()
            node_embeddings = model(batch)
            all_edges = torch.cat([batch.pos_edge_index.t(), batch.neg_edge_index.t()], dim=0)
            link_probs = model.predict_links(node_embeddings, all_edges)
            labels = torch.cat([
                torch.ones(batch.pos_edge_index.size(1)),
                torch.zeros(batch.neg_edge_index.size(1))
            ]).to(device)
            loss = criterion(link_probs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        val_auc = val_metrics['AUC']
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
        epoch_results.append({'epoch': epoch+1, 'loss': total_loss/len(train_loader), 'val_auc': val_auc})
        print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val AUC={val_auc:.4f}')
    # Save results to CSV
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'loss', 'val_auc'])
        writer.writeheader()
        for row in epoch_results:
            writer.writerow(row)
    # Print summary table
    print("\nTraining Summary:")
    print(f"{'Epoch':<6}{'Loss':<12}{'Val AUC':<10}")
    for row in epoch_results:
        print(f"{row['epoch']:<6}{row['loss']:<12.4f}{row['val_auc']:<10.4f}")

if __name__ == '__main__':
    main() 