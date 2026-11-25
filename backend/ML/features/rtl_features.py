import re


def extract_rtl_features(source: str) -> dict[str, int]:
    return {
        "modules": len(re.findall(r"\bmodule\b", source)),
        "ports": len(re.findall(r"\b(input|output|inout)\b", source)),
        "registers": len(re.findall(r"\breg\b", source)),
        "always_blocks": len(re.findall(r"\balways\b", source)),
    }
