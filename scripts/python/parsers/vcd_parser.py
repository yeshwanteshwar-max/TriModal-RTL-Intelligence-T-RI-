from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VCDSignal:
    identifier: str
    name: str
    width: int
    scope: str
    values: list[tuple[int, str]]
    signal_type: str = "wire"
    initial_value: str | None = None
    aliases: list[str] = field(default_factory=list)
    values_truncated: bool = False


class VCDSignals(dict[str, VCDSignal]):
    def __init__(self):
        super().__init__()
        self.timestamp_count = 0
        self.start_timestamp: int | None = None
        self.end_timestamp: int | None = None
        self.timescale: str | None = None
        self.values_truncated = False


def parse_vcd(
    path: str,
    max_signals: int = 5000,
    max_values: int = 500000,
) -> VCDSignals:
    vcd_path = Path(path)
    if not vcd_path.is_file():
        raise FileNotFoundError(f"VCD file not found: {vcd_path}")
    if max_signals <= 0 or max_values <= 0:
        raise ValueError("max_signals and max_values must be positive")

    scope: list[str] = []
    declarations: dict[str, tuple[str, int, str, str, list[str]]] = {}
    values: dict[str, list[tuple[int, str]]] = {}
    ignored_identifiers: set[str] = set()
    timestamp = 0
    last_timestamp: int | None = None
    timestamp_count = 0
    start_timestamp: int | None = None
    end_timestamp: int | None = None
    timescale: str | None = None
    value_count = 0
    values_truncated = False
    saw_content = False
    saw_enddefinitions = False
    in_text_directive = False
    text_directive: str | None = None
    text_parts: list[str] = []

    with vcd_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            saw_content = True
            is_value_change = False

            if in_text_directive:
                if text_directive == "timescale":
                    text_parts.append(line.replace("$end", "").strip())
                if "$end" in line:
                    if text_directive == "timescale":
                        timescale = " ".join(part for part in text_parts if part)
                    in_text_directive = False
                    text_directive = None
                continue
            if line.startswith("$timescale"):
                timescale_value = line[len("$timescale") :].replace("$end", "").strip()
                if "$end" in line:
                    timescale = timescale_value
                else:
                    in_text_directive = True
                    text_directive = "timescale"
                    text_parts = [timescale_value]
                continue
            if line.startswith(("$date", "$version", "$comment")):
                in_text_directive = "$end" not in line
                continue
            if line.startswith("$scope"):
                parts = line.split()
                if len(parts) < 4 or parts[-1] != "$end":
                    raise ValueError(f"Invalid $scope directive in {vcd_path}: {line}")
                scope.append(parts[2])
            elif line == "$upscope $end":
                if not scope:
                    raise ValueError(f"Unmatched $upscope directive in {vcd_path}")
                scope.pop()
            elif line.startswith("$var"):
                if saw_enddefinitions:
                    raise ValueError(f"$var directive after $enddefinitions in {vcd_path}")
                parts = line.split()
                if len(parts) < 6 or parts[-1] != "$end":
                    raise ValueError(f"Invalid $var directive in {vcd_path}: {line}")
                signal_type, width_text, identifier, name = parts[1:5]
                try:
                    width = int(width_text)
                except ValueError as error:
                    raise ValueError(f"Invalid signal width in {vcd_path}: {line}") from error
                if width <= 0 or not identifier:
                    raise ValueError(f"Invalid signal declaration in {vcd_path}: {line}")

                full_name = ".".join((*scope, name))
                if identifier in declarations:
                    previous_name, previous_width, previous_scope, previous_type, aliases = (
                        declarations[identifier]
                    )
                    if width != previous_width:
                        raise ValueError(
                            f"Aliased VCD signal {identifier!r} has inconsistent widths"
                        )
                    if full_name not in aliases and full_name != previous_name:
                        aliases.append(full_name)
                    declarations[identifier] = (
                        previous_name,
                        previous_width,
                        previous_scope,
                        previous_type,
                        aliases,
                    )
                elif len(declarations) < max_signals:
                    declarations[identifier] = (
                        full_name,
                        width,
                        ".".join(scope),
                        signal_type,
                        [],
                    )
                    values[identifier] = []
                else:
                    ignored_identifiers.add(identifier)
            elif line == "$enddefinitions $end":
                saw_enddefinitions = True
            elif line.startswith("#"):
                if not saw_enddefinitions:
                    raise ValueError(f"Timestamp before $enddefinitions in {vcd_path}")
                try:
                    timestamp = int(line[1:])
                except ValueError as error:
                    raise ValueError(f"Invalid VCD timestamp in {vcd_path}: {line}") from error
                if timestamp < 0:
                    raise ValueError(f"Negative VCD timestamp in {vcd_path}: {line}")
                if last_timestamp is not None and timestamp < last_timestamp:
                    raise ValueError(f"VCD timestamps are not monotonic in {vcd_path}")
                if timestamp != last_timestamp:
                    timestamp_count += 1
                if start_timestamp is None:
                    start_timestamp = timestamp
                end_timestamp = timestamp
                last_timestamp = timestamp
            elif line[0] in "01xXzZ":
                identifier = line[1:].strip()
                if not identifier:
                    raise ValueError(f"Invalid scalar value change in {vcd_path}: {line}")
                value = line[0].lower()
                event = (identifier, value)
                is_value_change = True
            elif line[0] in "bB":
                parts = line[1:].split()
                if len(parts) != 2 or not parts[0]:
                    raise ValueError(f"Invalid binary value change in {vcd_path}: {line}")
                identifier = parts[1]
                value = parts[0].lower()
                if any(bit not in "01xz" for bit in value):
                    raise ValueError(f"Invalid binary value in {vcd_path}: {line}")
                event = (identifier, value)
                is_value_change = True
            elif line[0] in "rR":
                parts = line[1:].split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid real value change in {vcd_path}: {line}")
                identifier = parts[1]
                value = parts[0]
                event = (identifier, value)
                is_value_change = True
            else:
                is_value_change = False

            if is_value_change:
                if not saw_enddefinitions:
                    raise ValueError(f"Value change before $enddefinitions in {vcd_path}")
                if start_timestamp is None:
                    start_timestamp = timestamp
                    end_timestamp = timestamp
                    timestamp_count = 1
                    last_timestamp = timestamp
                if identifier not in declarations and identifier not in ignored_identifiers:
                    raise ValueError(
                        f"Value change references undeclared identifier {identifier!r}"
                    )
                if identifier in declarations:
                    if value_count < max_values:
                        values[identifier].append((timestamp, value))
                        value_count += 1
                    else:
                        values_truncated = True

    if not saw_content:
        return VCDSignals()
    if not saw_enddefinitions:
        raise ValueError(f"Invalid VCD file; missing $enddefinitions: {vcd_path}")

    signals = VCDSignals()
    for identifier, (name, width, scope_name, signal_type, aliases) in declarations.items():
        signal_values = values[identifier]
        signals[identifier] = VCDSignal(
            identifier=identifier,
            name=name,
            width=width,
            scope=scope_name,
            values=signal_values,
            signal_type=signal_type,
            initial_value=signal_values[0][1] if signal_values else None,
            aliases=aliases,
            values_truncated=values_truncated,
        )
    signals.timestamp_count = timestamp_count
    signals.start_timestamp = start_timestamp
    signals.end_timestamp = end_timestamp
    signals.timescale = timescale
    signals.values_truncated = values_truncated
    return signals


def summarize_vcd(path: str) -> dict[str, object]:
    signals = parse_vcd(path)
    signal_summaries = {}
    all_transitions = 0
    for signal in signals.values():
        transitions = max(0, len(signal.values) - 1)
        binary = [value for _, value in signal.values if set(value) <= set("01")]
        ones = sum(value.count("1") for value in binary)
        total_bits = sum(len(value) for value in binary)
        unknown = sum(value.count("x") for _, value in signal.values)
        high_z = sum(value.count("z") for _, value in signal.values)
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
