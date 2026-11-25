import torch


def concatenate_fusion(features: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(list(features.values()), dim=-1)
