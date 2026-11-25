from torch.utils.data import Dataset
import torch
from pathlib import Path
import json
import random


class RTLDataset(Dataset):
    def __init__(self, rtl_features, vcd_features, netlist_features, labels):
        self.rtl_features = torch.as_tensor(rtl_features, dtype=torch.float32)
        self.vcd_features = torch.as_tensor(vcd_features, dtype=torch.float32)
        self.netlist_features = torch.as_tensor(netlist_features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.rtl_features[index],
            self.vcd_features[index],
            self.netlist_features[index],
            self.labels[index],
        )


def load_metadata_records(metadata_dir: str) -> list[dict]:
    directory = Path(metadata_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Metadata directory not found: {directory}")

    records = []
    seen = set()
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(f"Invalid metadata file {path}: {error}") from error

        required = {"sample_id", "design_id", "rtl_path", "label", "dataset_version"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"{path} missing fields: {sorted(missing)}")
        if record["sample_id"] in seen:
            raise ValueError(f"Duplicate sample_id: {record['sample_id']}")
        seen.add(record["sample_id"])

        rtl_path = Path(record["rtl_path"])
        if not rtl_path.is_file():
            raise FileNotFoundError(f"RTL source not found for {record['sample_id']}: {rtl_path}")
        for modality in ("vcd_path", "netlist_path"):
            value = record.get(modality)
            if value and not Path(value).is_file():
                raise FileNotFoundError(f"{modality} not found for {record['sample_id']}: {value}")
        records.append(record)
    return records


def reproducible_split(
    records: list[dict],
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train and validation ratios must leave a test split")

    rng = random.Random(seed)
    families = {}
    for record in records:
        family = record["design_id"].rsplit("_", 1)[0]
        families.setdefault(family, []).append(record)
    family_names = list(families)
    rng.shuffle(family_names)
    train_count = int(len(family_names) * train_ratio)
    validation_count = int(len(family_names) * validation_ratio)
    train_names = set(family_names[:train_count])
    validation_names = set(family_names[train_count:train_count + validation_count])

    splits = {"train": [], "validation": [], "test": []}
    for family_name, family_records in families.items():
        if family_name in train_names:
            splits["train"].extend(family_records)
        elif family_name in validation_names:
            splits["validation"].extend(family_records)
        else:
            splits["test"].extend(family_records)
    return splits
