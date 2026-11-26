import re
from dataclasses import dataclass

import numpy as np

from scripts.python.parsers.netlist_parser import netlist_to_graph


NETLIST_NODE_FEATURE_DIM = 16


@dataclass(frozen=True)
class NetlistGraphFeatures:
    node_features: np.ndarray
    edge_index: np.ndarray
    graph_features: dict[str, int | float]
    node_ids: tuple[str, ...]


def extract_netlist_features(source: str) -> dict[str, int]:
    return {"cells": len(re.findall(r"^\s*\w+\s+\w+\s*\(", source, re.MULTILINE))}


def extract_netlist_graph(
    path: str,
    module_name: str | None = None,
) -> NetlistGraphFeatures:
    """Convert Yosys JSON into GraphSAGE-ready node features and edge indices."""
    graph = netlist_to_graph(path, module_name=module_name)
    node_ids = tuple(sorted(graph.nodes()))
    node_index = {node: index for index, node in enumerate(node_ids)}
    max_degree = max((graph.degree(node) for node in node_ids), default=1)
    max_width = max((int(graph.nodes[node].get("width", 1)) for node in node_ids), default=1)
    features = np.zeros((len(node_ids), NETLIST_NODE_FEATURE_DIM), dtype=np.float32)

    for index, node in enumerate(node_ids):
        attributes = graph.nodes[node]
        kind = attributes.get("kind")
        if kind == "port":
            features[index, 0] = 1.0
            direction_index = {"input": 3, "output": 4, "inout": 5}.get(
                attributes.get("direction")
            )
            if direction_index is not None:
                features[index, direction_index] = 1.0
        elif kind == "cell":
            features[index, 1] = 1.0
            cell_type = str(attributes.get("cell_type", "unknown")).lower()
            features[index, 6] = float(attributes.get("sequential", False))
            if any(token in cell_type for token in ("mux", "pmux")):
                family_index = 9
            elif any(token in cell_type for token in ("add", "sub", "mul", "div", "alu")):
                family_index = 8
            elif any(token in cell_type for token in ("eq", "ne", "gt", "lt", "cmp")):
                family_index = 10
            elif any(token in cell_type for token in ("mem", "ram", "rom")):
                family_index = 11
            elif any(token in cell_type for token in ("shift", "shl", "shr")):
                family_index = 12
            elif any(token in cell_type for token in ("and", "or", "xor", "not", "logic")):
                family_index = 7
            else:
                family_index = 13
            features[index, family_index] = 1.0
        else:
            features[index, 2] = 1.0

        features[index, 14] = graph.degree(node) / max_degree
        width = max(0, int(attributes.get("width", 1)))
        features[index, 15] = np.log1p(width) / np.log1p(max_width)

    edges = sorted(graph.edges())
    edge_index = np.asarray(
        [
            [node_index[source] for source, _ in edges],
            [node_index[target] for _, target in edges],
        ],
        dtype=np.int64,
    ).reshape(2, -1)
    node_types = [graph.nodes[node].get("kind") for node in node_ids]
    cells = [graph.nodes[node] for node in node_ids if graph.nodes[node].get("kind") == "cell"]
    ports = [graph.nodes[node] for node in node_ids if graph.nodes[node].get("kind") == "port"]
    graph_features = {
        "node_count": len(node_ids),
        "edge_count": graph.number_of_edges(),
        "port_count": node_types.count("port"),
        "input_port_count": sum(port.get("direction") == "input" for port in ports),
        "output_port_count": sum(port.get("direction") == "output" for port in ports),
        "cell_count": len(cells),
        "sequential_cell_count": sum(bool(cell.get("sequential")) for cell in cells),
        "net_count": node_types.count("net"),
        "mean_degree": (
            sum(graph.degree(node) for node in node_ids) / len(node_ids) if node_ids else 0.0
        ),
        "maximum_degree": max((graph.degree(node) for node in node_ids), default=0),
    }
    return NetlistGraphFeatures(features, edge_index, graph_features, node_ids)


def extract_yosys_netlist_features(path: str, module_name: str | None = None) -> dict[str, int | float]:
    return extract_netlist_graph(path, module_name=module_name).graph_features
