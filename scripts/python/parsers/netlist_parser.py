import re
from pathlib import Path
import networkx as nx


def parse_netlist(path: str) -> dict[str, list[str]]:
    source = Path(path).read_text(encoding="utf-8")
    instances = []
    for line in source.splitlines():
        match = re.match(r"\s*(\w+)\s+\w+\s*\(", line)
        if match:
            instances.append(match.group(1))
    return {"cells": instances}


def validate_netlist(path: str) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Netlist JSON not found: {source}")
    import json
    data = json.loads(source.read_text(encoding="utf-8"))
    modules = data.get("modules")
    if not isinstance(modules, dict):
        raise ValueError("Yosys JSON must contain a modules object")
    return data


def netlist_to_graph(path: str) -> nx.DiGraph:
    data = validate_netlist(path)
    graph = nx.DiGraph()
    for module_name, module in data["modules"].items():
        module_prefix = f"{module_name}/"

        ports = module.get("ports", {})
        for port_name, port in ports.items():
            node = f"{module_prefix}port:{port_name}"
            graph.add_node(
                node,
                kind="port",
                direction=port.get("direction", ""),
                width=len(port.get("bits", [])),
            )

        cells = module.get("cells", {})
        for cell_name, cell in cells.items():
            node = f"{module_prefix}cell:{cell_name}"
            graph.add_node(
                node,
                kind="cell",
                cell_type=cell.get("type", "unknown"),
                sequential=cell.get("type", "").startswith("$") and "dff" in cell.get("type", "").lower(),
            )

            for pin, net_ids in cell.get("connections", {}).items():
                if not isinstance(net_ids, list):
                    continue
                for net_id in net_ids:
                    net_node = f"{module_prefix}net:{net_id}"
                    graph.add_node(net_node, kind="net", width=len(net_ids))
                    graph.add_edge(node, net_node, port=pin)
