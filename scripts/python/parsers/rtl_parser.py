import re
from dataclasses import dataclass
from collections import Counter


@dataclass
class Module:
    name: str
    ports: list[str]


def parse_rtl(source: str) -> list[Module]:
    modules = []
    pattern = re.compile(r"\bmodule\s+(\w+)", re.DOTALL)
    for match in pattern.finditer(source):
        header_start = match.end()
        header_end = source.find(";", header_start)
        header = source[header_start:header_end] if header_end >= 0 else ""
        ports = [
            part.strip()
            for part in re.findall(r"(?:input|output|inout)\s+(?:wire|reg|logic)?\s*(?:\[[^\]]+\]\s*)?\w+", header)
        ]
        modules.append(Module(match.group(1), ports))
    return modules


HARDWARE_KEYWORDS = {
    "module", "endmodule", "input", "output", "inout", "wire", "reg", "logic",
    "always", "always_ff", "always_comb", "assign", "posedge", "negedge",
    "parameter", "localparam", "case", "endcase", "begin", "end", "if", "else",
}

RTL_TOKEN_PATTERN = re.compile(
    r"[A-Za-z_$][A-Za-z0-9_$]*|[0-9]+(?:'[bBoOdDhH][0-9A-Fa-f_xXzZ]+)?|"
    r"<<|>>|<=|>=|==|!=|===|!==|[!&|^~+*/<>()[\]{},;:.#@+-]"
)


def tokenize_rtl(source: str) -> list[str]:
    if not isinstance(source, str):
        raise TypeError("RTL source must be a string")
    tokens = RTL_TOKEN_PATTERN.findall(source)
    return [
        f"<KW:{token}>" if token in HARDWARE_KEYWORDS else token
        for token in tokens
        if token not in {"//", "/*", "*/"}
    ]


def extract_rtl_features(source: str) -> dict[str, int]:
    modules = parse_rtl(source)
    return {
        "modules": len(modules),
        "ports": sum(len(module.ports) for module in modules),
        "registers": len(re.findall(r"\b(?:reg|logic)\b", source)),
        "always_blocks": len(re.findall(r"\balways(?:_[a-z]+)?\b", source)),
        "assignments": len(re.findall(r"\bassign\b|<=|=", source)),
        "operators": sum(
            source.count(operator)
            for operator in ("+", "-", "*", "/", "&", "|", "^", "<<", ">>")
        ),
        "tokens": len(tokenize_rtl(source)),
    }
