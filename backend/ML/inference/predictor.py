import torch


def predict(model: torch.nn.Module, rtl, vcd, netlist) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(rtl, vcd, netlist).logits
