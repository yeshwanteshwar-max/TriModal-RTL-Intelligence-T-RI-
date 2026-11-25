from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class VCDSignal:
    identifier: str
    name: str
    width: int
    scope: str
    values: list[tuple[int, str]]


def parse_vcd(path: str, max_signals: int = 5000, max_values: int = 500000) -> dict[str, VCDSignal]:
    vcd_path = Path(path)
    if not vcd_path.is_file():
        raise FileNotFoundError(f"VCD file not found: {vcd_path}")

    scope: list[str] = []
    declarations: dict[str, tuple[str, int, str]] = {}
    values: dict[str, list[tuple[int, str]]] = {}
    timestamp = 0
    value_count = 0

    with vcd_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("$scope"):
                parts = line.split()
                if len(parts) >= 3:
                    scope.append(parts[2])
            elif line == "$upscope $end":
                if scope:
                    scope.pop()
            elif line.startswith("$var"):
                parts = line.split()
                if len(parts) >= 6:
                    width = int(parts[2]) if parts[2].isdigit() else 1
                    identifier = parts[3]
                    name = parts[4]
                    declarations[identifier] = (name, width, ".".join(scope))
                    values[identifier] = []
                    if len(declarations) >= max_signals:
                        break
            elif line.startswith("#"):
                timestamp = int(line[1:])
            elif line and line[0] in "01xXzZ":
                if len(line) == 1:
                    continue
                identifier = line[1:].strip()
                if identifier in values and value_count < max_values:
                    values[identifier].append((timestamp, line[0]))
                    value_count += 1
            elif line.startswith("b") or line.startswith("B"):
                parts = line[1:].split()
                if len(parts) != 2:
                    continue
                identifier = parts[1]
                if identifier in values and value_count < max_values:
                    values[identifier].append((timestamp, parts[0]))
                    value_count += 1

    signals: dict[str, VCDSignal] = {}
    for identifier, (name, width, scope_name) in declarations.items():
        full_name = f"{scope_name}.{name}" if scope_name else name
        signals[identifier] = VCDSignal(
            identifier=identifier,
            name=full_name,
            width=width,
            scope=scope_name,
            values=values.get(identifier, []),
        )
    return signals


def summarize_vcd(path: str) -> dict[str, object]:
    signals = parse_vcd(path)
    signal_summaries = {}
    all_transitions = 0
    for identifier, signal in signals.items():
        transitions = max(0, len(signal.values) - 1)
        binary = [value for _, value in signal.values if set(value) <= set("01")]
        ones = sum(value.count("1") for value in binary)
        total_bits = sum(len(value) for value in binary)
        unknown = sum(value.count("x") + value.count("X") for _, value in signal.values)
        high_z = sum(value.count("z") + value.count("Z") for _, value in signal.values)
        signal_summaries[signal.name] = {
            "width": signal.width,
            "transitions": transitions,
            "high_ratio": (ones / total_bits) if total_bits else 0.0,
            "unknown": unknown,
            "high_z": high_z,
        }
        all_transitions += transitions
    return {
        "signal_count": len(signals),
        "transitions": all_transitions,
        "signals": signal_summaries,
    }
