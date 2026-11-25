def select_action(metrics: dict[str, float]) -> str:
    if metrics.get("timing_slack", 0) < 0:
        return "pipeline"
    if metrics.get("area", 0) > 100:
        return "share_logic"
    return "gate_clock"
