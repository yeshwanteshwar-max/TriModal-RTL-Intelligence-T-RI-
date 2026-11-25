def summarize_prediction(value: float) -> dict[str, float | str]:
    return {"score": value, "recommendation": "review" if value > 0.5 else "acceptable"}
