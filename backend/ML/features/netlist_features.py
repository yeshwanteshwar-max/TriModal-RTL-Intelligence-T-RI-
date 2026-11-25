import re


def extract_netlist_features(source: str) -> dict[str, int]:
    return {"cells": len(re.findall(r"^\s*\w+\s+\w+\s*\(", source, re.MULTILINE))}
