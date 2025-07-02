import numpy as np

class TemporalGraphDataset:
    def __init__(self, snapshots, prediction_horizon=1):
        self.snapshots = snapshots
        self.prediction_horizon = prediction_horizon

    def create_training_samples(self):
        """Create training samples with temporal splits."""
        samples = []
        for i in range(len(self.snapshots) - self.prediction_horizon):
            historical = self.snapshots[max(0, i-5):i+1]
            target = self.snapshots[i + self.prediction_horizon]
            positive_edges = target['edges']
            negative_edges = self.sample_negative_edges(
                historical[-1]['graph'], positive_edges, ratio=1.0
            )
            samples.append({
                'historical_graphs': historical,
                'positive_edges': positive_edges,
                'negative_edges': negative_edges,
                'timestamp': target['timestamp']
            })
        return samples

    def sample_negative_edges(self, graph, positive_edges, ratio=1.0):
        """Sample negative edges that don't exist in current graph."""
        nodes = list(graph.nodes())
        neg_edges = []
        pos_edge_set = set(map(tuple, positive_edges))
        while len(neg_edges) < len(positive_edges) * ratio:
            src = np.random.choice(nodes)
            dst = np.random.choice(nodes)
            if src != dst and (src, dst) not in pos_edge_set:
                neg_edges.append([src, dst])
        return np.array(neg_edges) 