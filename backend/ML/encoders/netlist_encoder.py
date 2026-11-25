import torch


class NetlistEncoder(torch.nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()
        self.input = torch.nn.Linear(input_dim, hidden_dim)
        self.message = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if node_features.ndim != 2:
            raise ValueError("node_features must have shape [nodes, features]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        hidden = torch.relu(self.input(node_features))
        aggregated = torch.zeros_like(hidden)
        if edge_index.numel():
            source = hidden[edge_index[0]]
            aggregated.index_add_(0, edge_index[1], source)
        hidden = torch.relu(self.message(hidden + aggregated))

        if batch is None:
            pooled = hidden.mean(dim=0, keepdim=True)
        else:
            if batch_size is None:
                batch_size = int(batch.max().item()) + 1
            pooled = torch.zeros(
                (batch_size, hidden.shape[-1]),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            pooled.index_add_(0, batch, hidden)
        return self.output(pooled)
