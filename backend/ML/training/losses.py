import torch


def mse_loss() -> torch.nn.Module:
    return torch.nn.MSELoss()
