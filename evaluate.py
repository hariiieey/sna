import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch

def evaluate_model(model, test_loader, device, k_values=[10, 20, 50]):
    """Comprehensive evaluation with multiple metrics."""
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            # Skip batches with no positive or negative edges
            if (batch.pos_edge_index.dim() < 2 or batch.neg_edge_index.dim() < 2 or
                batch.pos_edge_index.size(1) == 0 or batch.neg_edge_index.size(1) == 0):
                continue
            node_embeddings = model(batch)
            all_edges = torch.cat([batch.pos_edge_index.t(), batch.neg_edge_index.t()], dim=0)
            link_probs = model.predict_links(node_embeddings, all_edges)
            labels = torch.cat([
                torch.ones(batch.pos_edge_index.size(1)),
                torch.zeros(batch.neg_edge_index.size(1))
            ])
            all_probs.extend(link_probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if len(all_labels) == 0:
        return {'AUC': 0, 'AP': 0, 'Precision@K': {}, 'Recall@K': {}}
    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    precision_k = {}
    recall_k = {}
    for k in k_values:
        precision_k[k], recall_k[k] = precision_recall_at_k(all_labels, all_probs, k)
    return {
        'AUC': auc,
        'AP': ap,
        'Precision@K': precision_k,
        'Recall@K': recall_k
    }

def precision_recall_at_k(labels, probs, k):
    """Calculate Precision@K and Recall@K."""
    sorted_indices = np.argsort(probs)[::-1]
    top_k_indices = sorted_indices[:k]
    precision_k = np.sum(np.array(labels)[top_k_indices]) / k
    total_positives = np.sum(labels)
    recall_k = np.sum(np.array(labels)[top_k_indices]) / total_positives if total_positives > 0 else 0
    return precision_k, recall_k 