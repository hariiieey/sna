import torch
from model import TemporalGraphTransformer

class ScalableGraphTransformer(TemporalGraphTransformer):
    def __init__(self, *args, use_gradient_checkpointing=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gradient_checkpointing = use_gradient_checkpointing
    def forward(self, data):
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                super().forward, data
            )
        else:
            return super().forward(data)

def create_temporal_mini_batches(graph_snapshots, batch_size=1000, num_neighbors=[10, 5]):
    from torch_geometric.loader import NeighborLoader
    batches = []
    for snapshot in graph_snapshots:
        loader = NeighborLoader(
            snapshot['graph'],
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=True
        )
        batches.extend(list(loader))
    return batches 