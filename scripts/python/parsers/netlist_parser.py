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


def netlist_to_graph(path: str, module_name: str | None = None) -> nx.DiGraph:
    data = validate_netlist(path)
    graph = nx.DiGraph()
    modules = data["modules"]
    if module_name is not None:
        if module_name not in modules:
            raise ValueError(f"Module {module_name!r} not found in netlist")
        selected_modules = {module_name: modules[module_name]}
    else:
        selected_modules = modules

    for selected_name, module in selected_modules.items():
        module_prefix = f"{selected_name}/"

        ports = module.get("ports", {})
        for port_name, port in ports.items():
            node = f"{module_prefix}port:{port_name}"
            bits = port.get("bits", [])
            graph.add_node(
                node,
                kind="port",
                direction=port.get("direction", ""),
                width=len(bits),
            )
            for bit in bits:
                net_node = f"{module_prefix}net:{bit}"
                graph.add_node(net_node, kind="net", width=1)
                direction = port.get("direction", "")
                if direction == "input":
                    graph.add_edge(node, net_node, port=port_name)
                elif direction == "output":
                    graph.add_edge(net_node, node, port=port_name)
                else:
                    graph.add_edge(node, net_node, port=port_name)
                    graph.add_edge(net_node, node, port=port_name)

        cells = module.get("cells", {})
        for cell_name, cell in cells.items():
            node = f"{module_prefix}cell:{cell_name}"
            cell_type = cell.get("type", "unknown")
            graph.add_node(
                node,
                kind="cell",
                cell_type=cell_type,
                sequential=any(token in cell_type.lower() for token in ("dff", "latch")),
            )

            directions = cell.get("port_directions", {})
            for pin, net_ids in cell.get("connections", {}).items():
                if not isinstance(net_ids, list):
                    continue
                direction = directions.get(pin, "")
                for net_id in net_ids:
                    net_node = f"{module_prefix}net:{net_id}"
                    graph.add_node(net_node, kind="net", width=1)
                    if direction == "input":
                        graph.add_edge(net_node, node, port=pin)
                    elif direction == "output":
                        graph.add_edge(node, net_node, port=pin)
                    else:
                        graph.add_edge(node, net_node, port=pin)
                        if direction == "inout":
                            graph.add_edge(net_node, node, port=pin)
    return graph
