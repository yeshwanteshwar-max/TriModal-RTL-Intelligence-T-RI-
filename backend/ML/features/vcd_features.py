from __future__ import annotations

from dataclasses import dataclass
import math
import re
from pathlib import Path

import numpy as np

from scripts.python.parsers.vcd_parser import VCDSignal, parse_vcd


VCD_SEQUENCE_FEATURES = (
    "normalized_time",
    "normalized_time_delta",
    "active_signal_ratio",
    "toggles_per_bit",
    "rising_edges_per_bit",
    "falling_edges_per_bit",
    "high_bit_ratio",
    "low_bit_ratio",
    "unknown_bit_ratio",
    "high_impedance_bit_ratio",
    "normalized_numeric_mean",
    "inferred_clock_high_ratio",
    "inferred_reset_active_ratio",
)
VCD_SEQUENCE_FEATURE_DIM = len(VCD_SEQUENCE_FEATURES)


@dataclass(frozen=True)
class VCDFeatureConfig:
    sequence_length: int = 256
    max_signals: int = 5000
    max_values: int = 500000
    sampling_period: int | None = None
    normalization: str = "none"

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.max_signals <= 0 or self.max_values <= 0:
            raise ValueError("max_signals and max_values must be positive")
        if self.sampling_period is not None and self.sampling_period <= 0:
            raise ValueError("sampling_period must be positive")
        if self.normalization not in {"none", "minmax", "zscore"}:
            raise ValueError("normalization must be 'none', 'minmax', or 'zscore'")


def _bits(value: str, width: int) -> str | None:
    if not value or any(bit not in "01xzXZ" for bit in value):
        return None
    value = value.lower()
    if len(value) < width:
        extension = value[0] if value[0] in "xz" else "0"
        value = extension * (width - len(value)) + value
    return value[-width:]


def _numeric_value(value: str) -> float | None:
    try:
        if value.lower().startswith(("x", "z")):
            return None
        if any(bit not in "01" for bit in value):
            numeric = float(value)
        else:
            numeric = float(int(value, 2))
        return numeric if math.isfinite(numeric) else None
    except (ValueError, OverflowError):
        return None


def _signal_metrics(
    signal: VCDSignal,
    duration: int,
    simulation_end: int,
) -> dict[str, int | float | str | None]:
    transitions = 0
    rising_edges = 0
    falling_edges = 0
    numeric_values: list[float] = []
    timed_states = {"high": 0.0, "low": 0.0, "unknown": 0.0, "high_z": 0.0}
    sampled_states = {"high": 0.0, "low": 0.0, "unknown": 0.0, "high_z": 0.0}

    for index, (timestamp, value) in enumerate(signal.values):
        numeric = _numeric_value(value)
        if numeric is not None:
            numeric_values.append(numeric)
        if index:
            previous = signal.values[index - 1][1]
            old_bits = _bits(previous, signal.width)
            new_bits = _bits(value, signal.width)
            if previous != value:
                transitions += 1
            if old_bits is not None and new_bits is not None:
                rising_edges += sum(old == "0" and new == "1" for old, new in zip(old_bits, new_bits))
                falling_edges += sum(old == "1" and new == "0" for old, new in zip(old_bits, new_bits))

        bits = _bits(value, signal.width)
        if bits is None:
            continue
        next_timestamp = (
            signal.values[index + 1][0]
            if index + 1 < len(signal.values)
            else max(timestamp, simulation_end)
        )
        dwell = max(0, next_timestamp - timestamp)
        state_counts = timed_states if dwell else sampled_states
        if dwell:
            state_counts["high"] += bits.count("1") * dwell
            state_counts["low"] += bits.count("0") * dwell
            state_counts["unknown"] += bits.count("x") * dwell
            state_counts["high_z"] += bits.count("z") * dwell
        else:
            state_counts["high"] += bits.count("1")
            state_counts["low"] += bits.count("0")
            state_counts["unknown"] += bits.count("x")
            state_counts["high_z"] += bits.count("z")

    state_durations = timed_states if sum(timed_states.values()) else sampled_states
    total_state = sum(state_durations.values())
    denominator = total_state or 1.0
    toggle_count = rising_edges + falling_edges
    rate = toggle_count / duration if duration > 0 else 0.0
    return {
        "name": signal.name,
        "aliases": signal.aliases,
        "signal_type": signal.signal_type,
        "width": signal.width,
        "initial_value": signal.initial_value,
        "event_count": len(signal.values),
        "transition_count": transitions,
        "rising_edge_count": rising_edges,
        "falling_edge_count": falling_edges,
        "toggle_count": toggle_count,
        "toggle_rate": rate,
        "high_state_percentage": state_durations["high"] / denominator,
        "low_state_percentage": state_durations["low"] / denominator,
        "unknown_state_percentage": state_durations["unknown"] / denominator,
        "high_impedance_percentage": state_durations["high_z"] / denominator,
        "average_value": sum(numeric_values) / len(numeric_values) if numeric_values else None,
        "minimum_value": min(numeric_values) if numeric_values else None,
        "maximum_value": max(numeric_values) if numeric_values else None,
        "value_change_frequency": transitions / duration if duration > 0 else 0.0,
        "activity_ratio": toggle_count / (signal.width * duration) if duration > 0 else 0.0,
        "values_truncated": signal.values_truncated,
    }


def extract_vcd_features(
    path: str,
    max_signals: int = 5000,
    max_values: int = 500000,
) -> dict[str, object]:
    signals = parse_vcd(path, max_signals=max_signals, max_values=max_values)
    all_timestamps = [timestamp for signal in signals.values() for timestamp, _ in signal.values]
    timestamp_count = signals.timestamp_count
    simulation_start = (
        signals.start_timestamp
        if signals.start_timestamp is not None
        else min(all_timestamps, default=0)
    )
    simulation_end = (
        signals.end_timestamp
        if signals.end_timestamp is not None
        else max(all_timestamps, default=0)
    )
    duration = max(0, simulation_end - simulation_start)
    details = {
        signal.name: _signal_metrics(signal, duration, simulation_end)
        for signal in signals.values()
    }
    rates = [float(item["toggle_rate"]) for item in details.values()]
    widths = sum(signal.width for signal in signals.values())
    total_toggles = sum(int(item["toggle_count"]) for item in details.values())
    vector_toggles = sum(
        int(details[signal.name]["toggle_count"])
        for signal in signals.values()
        if signal.width > 1
    )
    clock_signals = [
        signal
        for signal in signals.values()
        if re.search(r"(^|[._])(?:clk|clock)(?:$|[_\d])", signal.name, re.IGNORECASE)
    ]
    reset_signals = [
        signal
        for signal in signals.values()
        if re.search(r"(?:^|[._])(?:rst|reset)", signal.name, re.IGNORECASE)
    ]
    scalar_count = sum(signal.width == 1 for signal in signals.values())
    event_count = sum(len(signal.values) for signal in signals.values())
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        lines = sum(1 for _ in handle)

    return {
        "lines": lines,
        "transitions": event_count,
        "signal_count": len(signals),
        "scalar_signal_count": scalar_count,
        "vector_signal_count": len(signals) - scalar_count,
        "event_count": event_count,
        "timestamp_count": timestamp_count,
        "simulation_duration": duration,
        "timescale": signals.timescale,
        "duration_units": signals.timescale or "VCD timestamp ticks",
        "average_signal_toggle_rate": sum(rates) / len(rates) if rates else 0.0,
        "maximum_signal_toggle_rate": max(rates, default=0.0),
        "minimum_signal_toggle_rate": min(rates, default=0.0),
        "aggregate_activity": total_toggles / (widths * duration) if widths and duration else 0.0,
        "bus_activity": vector_toggles / total_toggles if total_toggles else 0.0,
        "clock_like_candidates": [
            {"name": signal.name, "inferred": True, "heuristic": "signal name contains clk/clock"}
            for signal in clock_signals
        ],
        "reset_like_candidates": [
            {
                "name": signal.name,
                "inferred": True,
                "active_low_inferred": signal.name.lower().endswith("_n"),
                "heuristic": "signal name contains rst/reset; polarity is inferred from _n suffix",
            }
            for signal in reset_signals
        ],
        "signal_activity_distribution": sorted(rates),
        "values_truncated": signals.values_truncated,
        "signals": details,
    }


def extract_vcd_sequence(
    path: str,
    sequence_length: int = 256,
    max_signals: int = 5000,
    max_values: int = 500000,
    sampling_period: int | None = None,
    normalization: str = "none",
) -> np.ndarray:
    config = VCDFeatureConfig(
        sequence_length=sequence_length,
        max_signals=max_signals,
        max_values=max_values,
        sampling_period=sampling_period,
        normalization=normalization,
    )
    signals = parse_vcd(path, max_signals=config.max_signals, max_values=config.max_values)
    sequence = np.zeros((config.sequence_length, VCD_SEQUENCE_FEATURE_DIM), dtype=np.float32)
    if not signals:
        if config.normalization == "minmax":
            minimum = sequence.min(axis=0)
            scale = sequence.max(axis=0) - minimum
            sequence = np.divide(
                sequence - minimum,
                scale,
                out=np.zeros_like(sequence),
                where=scale > 0,
            )
        elif config.normalization == "zscore":
            mean = sequence.mean(axis=0)
            standard_deviation = sequence.std(axis=0)
            sequence = np.divide(
                sequence - mean,
                standard_deviation,
                out=np.zeros_like(sequence),
                where=standard_deviation > 0,
            )
        return sequence.astype(np.float32, copy=False)

    events: dict[int, list[tuple[str, str]]] = {}
    for identifier, signal in signals.items():
        for timestamp, value in signal.values:
            events.setdefault(timestamp, []).append((identifier, value))
    timestamps = sorted(events)
    if not timestamps:
        return sequence

    first_timestamp = (
        signals.start_timestamp if signals.start_timestamp is not None else timestamps[0]
    )
    simulation_end = (
        signals.end_timestamp if signals.end_timestamp is not None else timestamps[-1]
    )
    duration = max(0, simulation_end - first_timestamp)
    signal_width = sum(signal.width for signal in signals.values())
    clock_ids = {
        identifier
        for identifier, signal in signals.items()
        if re.search(r"(^|[._])(?:clk|clock)(?:$|[_\d])", signal.name, re.IGNORECASE)
    }
    reset_ids = {
        identifier
        for identifier, signal in signals.items()
        if re.search(r"(?:^|[._])(?:rst|reset)", signal.name, re.IGNORECASE)
    }
    states: dict[str, str] = {}
    timestamp_buckets: dict[int, list[int]] = {}
    for timestamp in timestamps:
        if config.sampling_period is not None:
            bucket = (timestamp - first_timestamp) // config.sampling_period
        elif duration:
            bucket = round((timestamp - first_timestamp) * (config.sequence_length - 1) / duration)
        else:
            bucket = 0
        bucket = min(config.sequence_length - 1, max(0, int(bucket)))
        timestamp_buckets.setdefault(bucket, []).append(timestamp)

    for step in range(config.sequence_length):
        changed_ids: set[str] = set()
        rising = 0
        falling = 0
        for timestamp in timestamp_buckets.get(step, []):
            for identifier, value in events[timestamp]:
                signal = signals[identifier]
                previous = states.get(identifier)
                previous_bits = _bits(previous, signal.width) if previous is not None else None
                current_bits = _bits(value, signal.width)
                if previous_bits is not None and current_bits is not None:
                    rising += sum(
                        old == "0" and new == "1"
                        for old, new in zip(previous_bits, current_bits)
                    )
                    falling += sum(
                        old == "1" and new == "0"
                        for old, new in zip(previous_bits, current_bits)
                    )
                if previous != value:
                    changed_ids.add(identifier)
                states[identifier] = value

        bit_counts = {"1": 0, "0": 0, "x": 0, "z": 0}
        numeric_values = []
        clock_high = []
        reset_active = []
        for identifier, value in states.items():
            signal = signals[identifier]
            bits = _bits(value, signal.width)
            if bits is not None:
                for bit in bit_counts:
                    bit_counts[bit] += bits.count(bit)
            numeric = _numeric_value(value)
            if numeric is not None:
                max_value = (2**signal.width) - 1
                numeric_values.append(
                    math.tanh(numeric)
                    if signal.signal_type == "real"
                    else numeric / max(1, max_value)
                )
            if identifier in clock_ids and bits and bits[-1] in "01":
                clock_high.append(float(bits[-1] == "1"))
            if identifier in reset_ids and bits and bits[-1] in "01":
                active_low = signal.name.lower().endswith("_n")
                reset_active.append(float(bits[-1] == ("0" if active_low else "1")))

        state_width = sum(bit_counts.values()) or 1
        denominator = max(1, signal_width)
        delta = 1.0 / max(1, config.sequence_length - 1)
        sequence[step] = (
            step / max(1, config.sequence_length - 1),
            delta,
            len(changed_ids) / max(1, len(signals)),
            (rising + falling) / denominator,
            rising / denominator,
            falling / denominator,
            bit_counts["1"] / state_width,
            bit_counts["0"] / state_width,
            bit_counts["x"] / state_width,
            bit_counts["z"] / state_width,
            sum(numeric_values) / len(numeric_values) if numeric_values else 0.0,
            sum(clock_high) / len(clock_high) if clock_high else 0.0,
            sum(reset_active) / len(reset_active) if reset_active else 0.0,
        )
    return sequence


def window_vcd_sequence(
    sequence: np.ndarray,
    window_size: int,
    overlap: float = 0.0,
) -> np.ndarray:
    if sequence.ndim != 2:
        raise ValueError("sequence must have shape [time, features]")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if len(sequence) <= window_size:
        padded = np.zeros((window_size, sequence.shape[1]), dtype=sequence.dtype)
        padded[: len(sequence)] = sequence
        return padded[None, ...]

    stride = max(1, int(window_size * (1.0 - overlap)))
    starts = list(range(0, len(sequence) - window_size + 1, stride))
    if starts[-1] != len(sequence) - window_size:
        starts.append(len(sequence) - window_size)
    return np.stack([sequence[start : start + window_size] for start in starts])
