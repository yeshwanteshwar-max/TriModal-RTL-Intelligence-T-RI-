from .iverilog_runner import compile_and_run


def generate_waveform(source: str, testbench: str, vvp_output: str) -> None:
    compile_and_run(source, testbench, vvp_output)
