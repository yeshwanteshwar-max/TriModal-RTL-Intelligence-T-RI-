from collections.abc import Sequence
import math
from statistics import median

import torch


BEHAVIOR_CLASSES = (
    "combinational",
    "sequential",
    "counter",
    "fsm",
    "shift_register",
    "alu",
    "fifo",
    "decoder",
    "encoder",
    "comparator",
    "unknown",
)
ACTIVITY_CLASSES = ("low", "moderate", "high")


class VCDPredictionHeads(torch.nn.Module):
    """Optional task heads kept separate from the VCD sequence encoder."""

    def __init__(
        self,
        embedding_dim: int = 128,
        behavior_classes: int | None = len(BEHAVIOR_CLASSES),
        activity_classes: int | None = len(ACTIVITY_CLASSES),
        anomaly_classes: int | None = None,
        regression_targets: int = 0,
    ):
        super().__init__()
        if embedding_dim <= 0 or regression_targets < 0:
            raise ValueError("embedding_dim must be positive and regression_targets non-negative")
        for task_classes in (behavior_classes, activity_classes, anomaly_classes):
            if task_classes is not None and task_classes <= 0:
                raise ValueError("task class counts must be positive when enabled")
        self.embedding_dim = embedding_dim
        self.behavior = (
            torch.nn.Linear(embedding_dim, behavior_classes)
            if behavior_classes is not None
            else None
        )
        self.activity = (
            torch.nn.Linear(embedding_dim, activity_classes)
            if activity_classes is not None
            else None
        )
        self.anomaly = (
            torch.nn.Linear(embedding_dim, anomaly_classes)
            if anomaly_classes is not None
            else None
        )
        self.regression = (
            torch.nn.Linear(embedding_dim, regression_targets)
            if regression_targets
            else None
        )

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        if embedding.ndim != 2 or embedding.shape[-1] != self.embedding_dim:
            raise ValueError(f"embedding must have shape [batch, {self.embedding_dim}]")
        outputs = {}
        if self.behavior is not None:
            outputs["behavior"] = self.behavior(embedding)
        if self.activity is not None:
            outputs["activity"] = self.activity(embedding)
        if self.anomaly is not None:
            outputs["anomaly"] = self.anomaly(embedding)
        if self.regression is not None:
            outputs["regression"] = self.regression(embedding)
        return outputs


def classify_activity(
    aggregate_activity: float,
    low_threshold: float = 0.1,
    high_threshold: float = 0.5,
) -> str:
    if not all(math.isfinite(value) for value in (aggregate_activity, low_threshold, high_threshold)):
        raise ValueError("activity and thresholds must be finite")
    if aggregate_activity < 0:
        raise ValueError("aggregate_activity must be non-negative")
    if not 0.0 <= low_threshold < high_threshold:
        raise ValueError("activity thresholds must satisfy 0 <= low_threshold < high_threshold")
    if aggregate_activity < low_threshold:
        return "low"
    if aggregate_activity < high_threshold:
        return "moderate"
    return "high"


def detect_statistical_anomalies(
    summary: dict,
    reference_summaries: Sequence[dict] | None = None,
    z_threshold: float = 3.0,
    unknown_threshold: float = 0.2,
    high_impedance_threshold: float = 0.2,
) -> list[dict[str, str | float | bool]]:
    """Return heuristic/statistical anomaly candidates, not verified functional bugs."""
    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive")
    for threshold in (unknown_threshold, high_impedance_threshold):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("state thresholds must be in [0, 1]")

    findings: list[dict[str, str | float | bool]] = []
    activity = float(summary.get("aggregate_activity", 0.0))
    if reference_summaries:
        baseline = [
            float(item.get("aggregate_activity", 0.0))
            for item in reference_summaries
            if "aggregate_activity" in item
        ]
        if baseline:
            baseline_median = median(baseline)
            mad = median([abs(value - baseline_median) for value in baseline])
            robust_score = (
                0.6745 * (activity - baseline_median) / mad
                if mad
                else (z_threshold + 1.0 if activity != baseline_median else 0.0)
            )
            if abs(robust_score) > z_threshold:
                findings.append(
                    {
                        "type": "abnormal_activity",
                        "signal": "",
                        "score": robust_score,
                        "explanation": "Aggregate activity differs from the reference median.",
                        "verified": False,
                    }
                )

    for signal in summary.get("signals", {}).values():
        name = str(signal.get("name", ""))
        unknown = float(signal.get("unknown_state_percentage", 0.0))
        high_z = float(signal.get("high_impedance_percentage", 0.0))
        if unknown > unknown_threshold:
            findings.append(
                {
                    "type": "unexpected_unknown_state",
                    "signal": name,
                    "score": unknown,
                    "explanation": "Unknown-state occupancy exceeds the configured threshold.",
                    "verified": False,
                }
            )
        if high_z > high_impedance_threshold:
            findings.append(
                {
                    "type": "unexpected_high_impedance",
                    "signal": name,
                    "score": high_z,
                    "explanation": "High-impedance occupancy exceeds the configured threshold.",
                    "verified": False,
                }
            )
        if (
            int(signal.get("event_count", 0)) > 1
            and int(signal.get("transition_count", 0)) == 0
            and int(summary.get("simulation_duration", 0)) > 0
        ):
            findings.append(
                {
                    "type": "possible_stuck_signal",
                    "signal": name,
                    "score": 1.0,
                    "explanation": "The signal had repeated observations but no value changes.",
                    "verified": False,
                }
            )
    return findings
