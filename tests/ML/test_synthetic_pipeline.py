from scripts.python.data_generation.synthetic_generator import generate_dataset
from backend.ML.training.dataset import load_metadata_records, reproducible_split
from backend.ML.core.multimodal_model import MultimodalModel
import torch


def test_synthetic_dataset_and_split(tmp_path):
    records = generate_dataset(
        str(tmp_path),
        families=["half_adder", "mux", "counter"],
        samples_per_family=3,
    )
    assert len(records) == 9
    metadata_dir = tmp_path / "metadata"
    loaded = load_metadata_records(str(metadata_dir))
    splits = reproducible_split(loaded)
    assert sum(len(records) for records in splits.values()) == 9
    assert set(splits) == {"train", "validation", "test"}


def test_multimodal_model_missing_modality():
    model = MultimodalModel()
    rtl = torch.zeros((2, 128))
    vcd = torch.zeros((2, 128))
    netlist = torch.zeros((2, 128))
    masks = torch.tensor([[1, 1, 1], [1, 0, 1]])
    output = model(rtl, vcd, netlist, masks)
    assert output.embedding.shape == (2, 256)
    assert set(output.logits) == {"classification", "ppa", "optimization"}
    assert output.confidence.shape == (2,)
