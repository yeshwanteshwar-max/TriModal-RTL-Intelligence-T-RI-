#!/usr/bin/env python3
"""Generate VVP/VCD simulation artifacts and a synthesized JSON netlist."""

from pathlib import Path
import argparse

from simulation.iverilog_runner import compile_and_run
from synthesis.pyosys_runner import generate_netlist_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RTL simulation and synthesis flow")
    parser.add_argument(
        "--rtl",
        default="design/rtl/counter.v",
        help="RTL source file",
    )
    parser.add_argument(
        "--testbench",
        default="design/tb/counter_tb.v",
        help="Verilog testbench",
    )
    parser.add_argument(
        "--vvp",
        default="waveform/vvp/counter.vvp",
        help="Compiled VVP output",
    )
    parser.add_argument(
        "--vcd",
        default="waveform/vcd/raw/counter.vcd",
        help="VCD output (created by the testbench)",
    )
    parser.add_argument(
        "--netlist",
        default="netlist",
        help="Netlist output directory",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    vvp_output = project_root / args.vvp
    vcd_output = project_root / args.vcd
    rtl_source = project_root / args.rtl
    testbench = project_root / args.testbench

    vvp_output.parent.mkdir(parents=True, exist_ok=True)
    vcd_output.parent.mkdir(parents=True, exist_ok=True)
    compile_and_run(
        str(rtl_source),
        str(testbench),
        str(vvp_output),
        cwd=project_root,
    )
    artifacts = generate_netlist_artifacts(
        str(rtl_source),
        str(project_root / args.netlist),
    )

    print(f"VVP: {vvp_output}")
    print(f"VCD: {vcd_output}")
    for kind, path in artifacts.items():
        print(f"{kind.title()}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
