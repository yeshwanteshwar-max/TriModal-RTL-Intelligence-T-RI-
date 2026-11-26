import torch


class GraphSAGELayer(torch.nn.Module):
    """Mean-neighbor GraphSAGE update using native PyTorch operations."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.update = torch.nn.Linear(input_dim * 2, output_dim)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        node_count = node_features.shape[0]
        neighbors = torch.zeros_like(node_features)
        degrees = node_features.new_zeros((node_count, 1))
        if edge_index.numel():
            source, target = edge_index
            neighbors.index_add_(0, target, node_features[source])
            degrees.index_add_(
                0,
                target,
                node_features.new_ones((target.shape[0], 1)),
            )
        neighbors = neighbors / degrees.clamp_min(1.0)
        return torch.relu(self.update(torch.cat((node_features, neighbors), dim=-1)))


class NetlistEncoder(torch.nn.Module):
    """GraphSAGE encoder for typed Yosys netlist graphs."""

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        normalize_embedding: bool = True,
    ):
        super().__init__()
        if min(input_dim, hidden_dim, output_dim, num_layers) <= 0:
            raise ValueError("input_dim, hidden_dim, output_dim, and num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = input_dim
        self.normalize_embedding = normalize_embedding
        self.layers = torch.nn.ModuleList(
            [
                GraphSAGELayer(input_dim if index == 0 else hidden_dim, hidden_dim)
                for index in range(num_layers)
            ]
        )
        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.projection = torch.nn.Linear(hidden_dim, output_dim)
        self.empty_embedding = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if node_features.ndim != 2 or node_features.shape[-1] != self.input_dim:
            raise ValueError(f"node_features must have shape [nodes, {self.input_dim}]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        if batch is not None and (batch.ndim != 1 or batch.shape[0] != node_features.shape[0]):
            raise ValueError("batch must have shape [nodes]")

        edge_index = edge_index.to(device=node_features.device, dtype=torch.long)
        if edge_index.numel() and (
            torch.any(edge_index < 0) or torch.any(edge_index >= node_features.shape[0])
        ):
            raise ValueError("edge_index contains a node index outside node_features")
        if edge_index.numel():
            edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)

        if batch is not None:
            batch = batch.to(device=node_features.device, dtype=torch.long)
            if torch.any(batch < 0):
                raise ValueError("batch indices must be non-negative")
            inferred_batch_size = int(batch.max().item()) + 1 if batch.numel() else 0
            if batch_size is None:
                batch_size = inferred_batch_size
            elif batch_size < inferred_batch_size:
                raise ValueError("batch_size is smaller than the largest batch index")
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
        elif batch_size is not None and batch_size != 1:
            raise ValueError("batch_size may only exceed one when batch indices are provided")

        if node_features.shape[0] == 0:
            output = self.empty_embedding.unsqueeze(0)
            if batch_size is not None:
                output = output.expand(batch_size, -1)
            return output

        hidden = node_features
        for layer, norm in zip(self.layers, self.norms):
            hidden = self.dropout(norm(layer(hidden, edge_index)))
        node_embeddings = self.projection(hidden)

        if batch is None:
            pooled = node_embeddings.mean(dim=0, keepdim=True)
            counts = node_embeddings.new_tensor([node_embeddings.shape[0]])
            actual_batch_size = 1
        else:
            actual_batch_size = batch_size
            pooled = node_embeddings.new_zeros((actual_batch_size, node_embeddings.shape[-1]))
            pooled.index_add_(0, batch, node_embeddings)
            counts = node_embeddings.new_zeros((actual_batch_size,))
            counts.index_add_(0, batch, node_embeddings.new_ones((batch.shape[0],)))
            pooled = pooled / counts.clamp_min(1.0).unsqueeze(-1)

        empty_mask = counts == 0
        if empty_mask.any():
            pooled[empty_mask] = self.empty_embedding
        if self.normalize_embedding:
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return pooled
