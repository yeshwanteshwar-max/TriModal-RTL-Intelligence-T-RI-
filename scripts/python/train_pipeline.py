#!/usr/bin/env python3
"""Create synthetic data and run a small multimodal training smoke run."""

from pathlib import Path
import argparse
import random
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.synthetic_generator import generate_dataset
from backend.ML.core.multimodal_model import MultimodalModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the multimodal RTL model")
    parser.add_argument("--output-dir", default="backend/ML/datasets/synthetic")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples-per-family", type=int, default=4)
    args = parser.parse_args()

    records = generate_dataset(args.output_dir, samples_per_family=args.samples_per_family)
    random.seed(42)
    torch.manual_seed(42)

    count = len(records)
    rtl = torch.randn(count, 128)
    vcd = torch.randn(count, 128)
    netlist = torch.randn(count, 128)
    masks = torch.ones((count, 3))
    masks[:, 1] = 0
    labels = torch.tensor(
        [0 if "counter" not in record["design_id"] else 3 for record in records],
        dtype=torch.long,
    )

    model = MultimodalModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    classifier_loss = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        output = model(rtl, vcd, netlist, masks)
        loss = classifier_loss(output.logits["classification"], labels)
        loss.backward()
        optimizer.step()
        print(f"epoch={epoch + 1} loss={loss.item():.6f}")

    checkpoint_dir = Path("backend/ML/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "multimodal_smoke.pt"
    torch.save(model.state_dict(), checkpoint)
    print(f"checkpoint={checkpoint}")
    print(f"samples={count}")
    print("missing modality in smoke training: vcd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
