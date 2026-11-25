from pathlib import Path
import json
import re
import shutil
import subprocess
from collections import Counter

from pyosys import libyosys


def find_top_module(source_path: Path) -> str:
    source = source_path.read_text(encoding="utf-8")
    match = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)", source)
    if not match:
        raise ValueError(f"No Verilog module found in {source_path}")
    return match.group(1)


def run_yosys(source: str, output: str) -> None:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"RTL source not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = find_top_module(source_path)
    script = (
        f"read_verilog {source_path}; "
        f"synth -top {top}; "
        f"write_json {output_path}"
    )
    libyosys.run_pass(script)


def generate_netlist_artifacts(source: str, output_directory: str) -> dict[str, Path]:
    source_path = Path(source).resolve()
    base_directory = Path(output_directory).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"RTL source not found: {source_path}")

    top = find_top_module(source_path)
    base_name = top
    directories = {
        "raw": base_directory / "raw",
        "json": base_directory / "json",
        "verilog": base_directory / "verilog",
        "graphs": base_directory / "graphs",
        "reports": base_directory / "reports",
        "img": base_directory / "img",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "raw": directories["raw"] / f"{base_name}.rtlil",
        "json": directories["json"] / f"{base_name}.json",
        "verilog": directories["verilog"] / f"{base_name}_netlist.v",
        "graphs": directories["graphs"] / f"{base_name}.dot",
        "reports": directories["reports"] / f"{base_name}_report.txt",
    }

    existing_numbers = []
    image_pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)\.png$")
    for image_path in directories["img"].glob(f"{base_name}_*.png"):
        match = image_pattern.match(image_path.name)
        if match:
            existing_numbers.append(int(match.group(1)))
    image_number = max(existing_numbers, default=0) + 1
    artifacts["img"] = directories["img"] / f"{base_name}_{image_number}.png"

    script = (
        f"read_verilog {source_path}; "
        f"synth -top {top}; "
        f"write_rtlil {artifacts['raw']}; "
        f"write_json {artifacts['json']}; "
        f"write_verilog -norename {artifacts['verilog']}"
    )
    libyosys.run_pass(script)

    netlist = json.loads(artifacts["json"].read_text(encoding="utf-8"))
    module = netlist["modules"].get(top, {})
    cells = module.get("cells", {})
    cell_types = Counter(cell.get("type", "unknown") for cell in cells.values())
    ports = module.get("ports", {})
    wires = module.get("netnames", {})

    dot_lines = [
        f'digraph "{top}" {{',
        '  rankdir=LR;',
        '  graph [dpi=180, pad="0.25", nodesep="0.45", ranksep="0.65"];',
        '  graph [splines=ortho, concentrate=true];',
        '  node [fontname="Liberation Sans", fontsize=10];',
        '  edge [fontname="Liberation Sans", fontsize=8];',
    ]
    port_nets: dict[str, set[str]] = {}
    for port_name, port in ports.items():
        port_nets[port_name] = {
            str(bit) for bit in port.get("bits", []) if isinstance(bit, (str, int))
        }
        direction = port.get("direction", "port")
        dot_lines.append(
            f'  "{port_name}" [shape=box, style=filled, '
            f'fillcolor="#dbeafe", label="{direction}\\\\n{port_name}"];'
        )
    for cell_name, cell in cells.items():
        cell_type = cell.get("type", "unknown")
        dot_lines.append(
            f'  "{cell_name}" [shape=ellipse, style=filled, '
            f'fillcolor="#ecfeff", label="{cell_name}\\\\n{cell_type}"];'
        )

    cell_nets: dict[str, set[str]] = {}
    for cell_name, cell in cells.items():
        nets = set()
        for connections in cell.get("connections", {}).values():
            if isinstance(connections, list):
                nets.update(str(bit) for bit in connections if isinstance(bit, (str, int)))
        cell_nets[cell_name] = nets

    for port_name, nets in port_nets.items():
        direction = ports[port_name].get("direction", "")
        if direction == "input":
            for cell_name, cell_net_values in cell_nets.items():
                if nets & cell_net_values:
                    dot_lines.append(
                        f'  "{port_name}" -> "{cell_name}" [color="#2563eb"];'
                    )
        elif direction == "output":
            for cell_name, cell_net_values in cell_nets.items():
                if nets & cell_net_values:
                    dot_lines.append(
                        f'  "{cell_name}" -> "{port_name}" [color="#16a34a"];'
                    )

    cell_names = list(cells)
    for left_index, left_name in enumerate(cell_names):
        for right_name in cell_names[left_index + 1:]:
            if cell_nets[left_name] & cell_nets[right_name]:
                dot_lines.append(
                    f'  "{left_name}" -> "{right_name}" '
                    '[color="#64748b", style=dashed];'
                )
    dot_lines.append("}")
    artifacts["graphs"].write_text("\n".join(dot_lines) + "\n", encoding="utf-8")

    dot_executable = shutil.which("dot")
    if not dot_executable:
        raise RuntimeError("Graphviz 'dot' executable is not installed")
    subprocess.run(
        [
            dot_executable,
            "-Tpng",
            "-Gdpi=180",
            str(artifacts["graphs"]),
            "-o",
            str(artifacts["img"]),
        ],
        check=True,
    )

    report_lines = [
        "RTL Netlist Generation Report",
        "=============================",
        f"Source: {source_path}",
        f"Top module: {top}",
        f"Ports: {len(ports)}",
        f"Wires: {len(wires)}",
        f"Cells: {len(cells)}",
        "",
        "Cell types:",
    ]
    report_lines.extend(
        f"  {cell_type}: {count}"
        for cell_type, count in sorted(cell_types.items())
    )
    report_lines.extend(["", "Generated artifacts:", *(f"  {kind}: {path}" for kind, path in artifacts.items())])
    artifacts["reports"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return artifacts
