def optimization_reward(area_ratio: float, timing_slack: float) -> float:
    return -(area_ratio + max(0.0, -timing_slack))
