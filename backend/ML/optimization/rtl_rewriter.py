def rewrite_clock_gate(source: str) -> str:
    return source.replace("always @(posedge clk)", "always @(posedge clk)")
