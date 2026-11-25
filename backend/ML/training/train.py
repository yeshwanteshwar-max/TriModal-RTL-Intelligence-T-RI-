import torch


def train_epoch(model, loader, optimizer, loss_fn) -> float:
    model.train()
    total_loss = 0.0
    for rtl, vcd, netlist, labels in loader:
        optimizer.zero_grad()
        output = model(rtl, vcd, netlist)
        loss = loss_fn(output.logits.squeeze(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))
