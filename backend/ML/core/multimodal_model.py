from dataclasses import dataclass
import torch


@dataclass
class FusionOutput:
    logits: dict[str, torch.Tensor]
    embedding: torch.Tensor
    missing_modalities: list[str]
    confidence: torch.Tensor


class MultimodalModel(torch.nn.Module):
    def __init__(
        self,
        dimensions: tuple[int, int, int] = (128, 128, 128),
        fused_dim: int = 256,
        class_count: int = 11,
    ):
        super().__init__()
        rtl_dim, vcd_dim, netlist_dim = dimensions
        self.dimensions = dimensions
        self.project_rtl = torch.nn.Linear(rtl_dim, fused_dim)
        self.project_vcd = torch.nn.Linear(vcd_dim, fused_dim)
        self.project_netlist = torch.nn.Linear(netlist_dim, fused_dim)
        self.gates = torch.nn.Linear(fused_dim * 3, 3)
        self.norm = torch.nn.LayerNorm(fused_dim)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(fused_dim, class_count)
        self.ppa = torch.nn.Sequential(
            torch.nn.Linear(fused_dim, fused_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(fused_dim // 2, 5),
        )
        self.optimization = torch.nn.Linear(fused_dim, 11)

    def forward(
        self,
        rtl: torch.Tensor,
        vcd: torch.Tensor,
        netlist: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> FusionOutput:
        batch_size = rtl.shape[0]
        if rtl.shape[-1] != self.dimensions[0]:
            raise ValueError(f"RTL embedding must have dimension {self.dimensions[0]}")
        if vcd.shape[-1] != self.dimensions[1]:
            raise ValueError(f"VCD embedding must have dimension {self.dimensions[1]}")
        if netlist.shape[-1] != self.dimensions[2]:
            raise ValueError(f"Netlist embedding must have dimension {self.dimensions[2]}")

        if masks is None:
            masks = torch.ones((batch_size, 3), device=rtl.device)
        if masks.shape != (batch_size, 3):
            raise ValueError("masks must have shape [batch_size, 3]")

        projected = torch.stack(
            (
                self.project_rtl(rtl),
                self.project_vcd(vcd),
                self.project_netlist(netlist),
            ),
            dim=1,
        )
        masked = projected * masks.unsqueeze(-1)
        gate_logits = self.gates(masked.reshape(batch_size, -1))
        gate_logits = gate_logits.masked_fill(masks <= 0, float("-inf"))
        gates = torch.softmax(gate_logits, dim=-1)
        embedding = self.norm(torch.sum(projected * gates.unsqueeze(-1), dim=1))
        embedding = self.dropout(embedding)

        logits = {
            "classification": self.classifier(embedding),
            "ppa": self.ppa(embedding),
            "optimization": self.optimization(embedding),
        }
        missing = [
            name
            for index, name in enumerate(("rtl", "vcd", "netlist"))
            if bool((masks[:, index] == 0).all().item())
        ]
        confidence = torch.softmax(logits["optimization"], dim=-1).max(dim=-1).values
        return FusionOutput(logits, embedding, missing, confidence)
