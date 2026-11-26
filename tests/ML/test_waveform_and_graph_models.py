import json

import numpy as np
import pytest
import torch

from backend.ML.encoders.netlist_encoder import NetlistEncoder
from backend.ML.encoders.vcd_encoder import VCDEncoder
from backend.ML.features.netlist_features import (
    NETLIST_NODE_FEATURE_DIM,
    extract_netlist_graph,
)
from backend.ML.features.vcd_features import (
    VCD_SEQUENCE_FEATURE_DIM,
    extract_vcd_features,
    extract_vcd_sequence,
    window_vcd_sequence,
)
from backend.ML.heads.vcd_heads import (
    VCDPredictionHeads,
    classify_activity,
    detect_statistical_anomalies,
)
from scripts.python.parsers.vcd_parser import parse_vcd


VCD_TEXT = """$date today $end
$timescale 1ns $end
$scope module top $end
$var wire 1 ! clk $end
$var wire 1 ! clk_alias $end
$var reg 2 " bus [1:0] $end
$var reg 1 # rst_n $end
$enddefinitions $end
#0
$dumpvars
0!
b0 "
1#
$end
#5
1!
b1 "
#10
0!
b10 "
x#
#15
z#
"""


def _write_netlist(path):
    path.write_text(
        json.dumps(
            {
                "modules": {
                    "top": {
                        "attributes": {"top": "1"},
                        "ports": {
                            "a": {"direction": "input", "bits": [2]},
                            "y": {"direction": "output", "bits": [4]},
                        },
                        "cells": {
                            "gate": {
                                "type": "$_AND_",
                                "port_directions": {
                                    "A": "input",
                                    "B": "input",
                                    "Y": "output",
                                },
                                "connections": {
                                    "A": [2],
                                    "B": [3],
                                    "Y": [4],
                                },
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_vcd_parser_and_engineered_features(tmp_path):
    path = tmp_path / "counter.vcd"
    path.write_text(VCD_TEXT + "#20\n", encoding="utf-8")

    signals = parse_vcd(str(path))
    assert len(signals) == 3
    clock = next(signal for signal in signals.values() if signal.name == "top.clk")
    assert clock.aliases == ["top.clk_alias"]
    assert clock.initial_value == "0"
    assert clock.values[-1] == (10, "0")

    summary = extract_vcd_features(str(path))
    assert summary["event_count"] == 9
    assert summary["timestamp_count"] == 5
    assert summary["simulation_duration"] == 20
    assert summary["clock_like_candidates"][0]["inferred"] is True
    assert summary["signals"]["top.clk"]["high_state_percentage"] == pytest.approx(0.25)
    assert summary["signals"]["top.clk"]["low_state_percentage"] == pytest.approx(0.75)
    assert summary["signals"]["top.bus"]["width"] == 2
    assert summary["signals"]["top.bus"]["toggle_count"] == 3

    sequence = extract_vcd_sequence(str(path), sequence_length=16)
    assert sequence.shape == (16, VCD_SEQUENCE_FEATURE_DIM)
    assert sequence.dtype == np.float32
    assert np.isfinite(sequence).all()
    normalized = extract_vcd_sequence(str(path), sequence_length=16, normalization="minmax")
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    windows = window_vcd_sequence(sequence, window_size=6, overlap=0.5)
    assert windows.shape == (5, 6, VCD_SEQUENCE_FEATURE_DIM)


def test_empty_and_malformed_vcd_inputs(tmp_path):
    empty_path = tmp_path / "empty.vcd"
    empty_path.write_text("", encoding="utf-8")
    assert parse_vcd(str(empty_path)) == {}
    assert extract_vcd_sequence(str(empty_path)).shape == (256, VCD_SEQUENCE_FEATURE_DIM)

    malformed_path = tmp_path / "malformed.vcd"
    malformed_path.write_text("$scope module top $end\n", encoding="utf-8")
    with pytest.raises(ValueError, match="enddefinitions"):
        parse_vcd(str(malformed_path))


def test_vcd_real_values_and_event_limit(tmp_path):
    path = tmp_path / "real.vcd"
    text = VCD_TEXT.replace(
        "$enddefinitions $end",
        '$var real 1 ? voltage $end\n$enddefinitions $end',
    ).replace("#0\n", "#0\nr1.25 ?\n").replace("#10\n", "#10\nr2.5 ?\n")
    path.write_text(text, encoding="utf-8")

    summary = extract_vcd_features(str(path), max_values=1)
    assert summary["values_truncated"] is True
    assert summary["signals"]["top.voltage"]["average_value"] == 1.25
    assert summary["signals"]["top.voltage"]["minimum_value"] == 1.25


def test_vcd_encoder_masks_padded_timesteps_and_heads():
    torch.manual_seed(7)
    encoder = VCDEncoder(dropout=0.0).eval()
    batch = torch.randn(2, 8, VCD_SEQUENCE_FEATURE_DIM)
    batch[0, 5:] = 1000.0
    with torch.no_grad():
        batched_embedding = encoder(batch, lengths=torch.tensor([5, 8]))
        single_embedding = encoder(batch[:1, :5])
    assert batched_embedding.shape == (2, 128)
    assert torch.allclose(batched_embedding[:1], single_embedding, atol=1e-6)
    assert torch.allclose(batched_embedding.norm(dim=-1), torch.ones(2), atol=1e-6)

    heads = VCDPredictionHeads(regression_targets=3)
    predictions = heads(batched_embedding)
    assert predictions["behavior"].shape == (2, 11)
    assert predictions["activity"].shape == (2, 3)
    assert predictions["regression"].shape == (2, 3)
    assert classify_activity(0.2) == "moderate"
    assert classify_activity(0.7, low_threshold=0.2, high_threshold=0.6) == "high"
    with pytest.raises(ValueError):
        classify_activity(0.2, low_threshold=0.5, high_threshold=0.4)

    anomalies = detect_statistical_anomalies(
        {
            "aggregate_activity": 0.9,
            "simulation_duration": 10,
            "signals": {
                "stuck": {
                    "name": "stuck",
                    "event_count": 2,
                    "transition_count": 0,
                    "unknown_state_percentage": 0.0,
                    "high_impedance_percentage": 0.0,
                }
            },
        },
        reference_summaries=[{"aggregate_activity": 0.1}] * 5,
    )
    assert {finding["type"] for finding in anomalies} == {
        "abnormal_activity",
        "possible_stuck_signal",
    }


def test_yosys_graph_features_and_graphsage_batching(tmp_path):
    path = tmp_path / "netlist.json"
    _write_netlist(path)
    graph_features = extract_netlist_graph(str(path), module_name="top")
    assert graph_features.node_features.shape == (
        graph_features.graph_features["node_count"],
        NETLIST_NODE_FEATURE_DIM,
    )
    assert graph_features.edge_index.shape[0] == 2
    assert graph_features.graph_features["cell_count"] == 1
    assert graph_features.graph_features["input_port_count"] == 1

    nodes = torch.as_tensor(graph_features.node_features)
    edges = torch.as_tensor(graph_features.edge_index)
    batch = torch.cat((torch.zeros(len(nodes)), torch.ones(len(nodes)))).long()
    batched_nodes = torch.cat((nodes, nodes))
    batched_edges = torch.cat((edges, edges + len(nodes)), dim=1)
    model = NetlistEncoder(output_dim=32, dropout=0.0)
    embedding = model(batched_nodes, batched_edges, batch=batch, batch_size=2)
    assert embedding.shape == (2, 32)
    assert torch.isfinite(embedding).all()
    embedding.sum().backward()
    assert model.layers[0].update.weight.grad is not None
