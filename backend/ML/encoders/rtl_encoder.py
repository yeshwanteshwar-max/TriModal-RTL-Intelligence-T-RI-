import torch


class RTLEncoder(torch.nn.Module):
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)
