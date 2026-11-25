`timescale 1ns/1ps

module counter_tb;
    reg clk;
    reg rst_n;
    reg enable;
    wire [7:0] count;
    integer errors;
    integer i;

    counter dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .count(count)
    );

    always #5 clk = ~clk;

    task check_count;
        input [7:0] expected;
        begin
            if (count !== expected) begin
                $display("FAIL: count=%0d, expected=%0d at time %0t",
                         count, expected, $time);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        enable = 1'b0;
        errors = 0;

        $dumpfile("waveform/vcd/raw/counter.vcd");
        $dumpvars(0, counter_tb);

        #12;
        check_count(8'd0);

        rst_n = 1'b1;
        enable = 1'b1;

        for (i = 1; i <= 10; i = i + 1) begin
            @(posedge clk);
            #1;
            check_count(i);
        end

        enable = 1'b0;
        @(posedge clk);
        #1;
        check_count(8'd10);

        @(posedge clk);
        #1;
        check_count(8'd10);

        rst_n = 1'b0;
        #1;
        check_count(8'd0);

        if (errors == 0)
            $display("PASS: counter reset, increment, and hold tests passed");
        else
            $display("FAIL: %0d counter test(s) failed", errors);

        $finish;
    end
endmodule
