from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit(self, dataset, epochs: int = 1) -> list[float]:
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        from .train import train_epoch
        losses = []
        for _ in range(epochs):
            losses.append(train_epoch(self.model, loader, self.optimizer, self.loss_fn))
        return losses
