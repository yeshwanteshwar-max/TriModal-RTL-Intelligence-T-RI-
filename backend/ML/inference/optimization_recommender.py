def recommend(objective: str, score: float) -> str:
    if objective == "area":
        return "Reduce combinational fanout and merge redundant logic"
    if objective == "timing":
        return "Add or move pipeline registers"
    if objective == "power":
        return "Gate low-activity signal paths"
    return "Analyze the design and select an optimization target"
