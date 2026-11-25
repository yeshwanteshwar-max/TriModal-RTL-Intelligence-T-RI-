import torch


class VCDEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.projection = torch.nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("VCD features must have shape [batch, time, signals]")
        if lengths is not None:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                features,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(features)
        return self.projection(hidden[-1])
