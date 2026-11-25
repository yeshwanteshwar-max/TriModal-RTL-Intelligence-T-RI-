#!/usr/bin/env python3
"""Generate deterministic synthetic RTL designs for pipeline validation."""

from pathlib import Path
import argparse
import json
from textwrap import dedent


TEMPLATES = {
    "half_adder": """
module {name}(
    input  wire a,
    input  wire b,
    output wire sum,
    output wire carry
);
assign sum = a ^ b;
assign carry = a & b;
endmodule
""",
    "full_adder": """
module {name}(
    input  wire a,
    input  wire b,
    input  wire cin,
    output wire sum,
    output wire carry
);
assign sum = a ^ b ^ cin;
assign carry = (a & b) | (cin & (a ^ b));
endmodule
""",
    "mux": """
module {name}(
    input  wire [{width}-1:0] a,
    input  wire [{width}-1:0] b,
    input  wire sel,
    output wire [{width}-1:0] y
);
assign y = sel ? b : a;
endmodule
""",
    "decoder": """
module {name}(
    input  wire [{address_width}-1:0] address,
    output reg  [{output_width}-1:0] decoded
);
integer i;
always @* begin
    decoded = {output_width}'b0;
    decoded[address] = 1'b1;
end
endmodule
""",
    "comparator": """
module {name}(
    input  wire [{width}-1:0] a,
    input  wire [{width}-1:0] b,
    output wire equal,
    output wire greater
);
assign equal = (a == b);
assign greater = (a > b);
endmodule
""",
    "counter": """
module {name}(
    input  wire clk,
    input  wire rst_n,
    input  wire enable,
    output reg [{width}-1:0] count
);
always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        count <= {width}'b0;
    else if (enable)
        count <= count + 1'b1;
end
endmodule
""",
}


def generate_sample(
    family: str,
    index: int,
    width: int,
) -> tuple[str, str]:
    if family not in TEMPLATES:
        raise ValueError(f"Unknown design family: {family}")
    name = f"{family}_{width}_{index}"
    source = dedent(
        TEMPLATES[family].format(
            name=name,
            width=width,
            address_width=max(1, width // 2),
            output_width=(1 << max(1, width // 2)),
        )
    ).strip() + "\n"
    return name, source


def generate_dataset(output_dir: str, families: list[str] | None = None, samples_per_family: int = 4) -> list[dict]:
    base = Path(output_dir)
    rtl_dir = base / "rtl"
    labels_dir = base / "labels"
    metadata_dir = base / "metadata"
    for directory in (rtl_dir, labels_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selected = families or list(TEMPLATES)
    records = []
    for family in selected:
        for index in range(samples_per_family):
            width = 2 + index * 2
            name, source = generate_sample(family, index, width)
            rtl_path = rtl_dir / f"{name}.v"
            rtl_path.write_text(source, encoding="utf-8")
            record = {
                "sample_id": name,
                "design_id": f"{family}_{width}",
                "rtl_path": str(rtl_path),
                "vcd_path": None,
                "netlist_path": None,
                "label": {
                    "design_class": "counter" if family == "counter" else "combinational",
                    "optimization": "no optimization",
                },
                "synthesis_metrics": {},
                "equivalence": None,
                "provenance": "deterministic synthetic generator",
                "dataset_version": "1",
            }
            record_path = labels_dir / f"{name}.json"
            record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            records.append(record)
            (metadata_dir / f"{name}.json").write_text(
                json.dumps(record, indent=2) + "\n",
                encoding="utf-8",
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic RTL samples")
    parser.add_argument("output_dir")
    parser.add_argument("--samples-per-family", type=int, default=4)
    parser.add_argument("--families", nargs="*", choices=list(TEMPLATES))
    args = parser.parse_args()
    records = generate_dataset(args.output_dir, args.families, args.samples_per_family)
    print(f"Generated {len(records)} synthetic RTL samples in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
