`timescale 1ns / 1ps
module tb_e2_pe_row();
    localparam int DATA_WIDTH = 16;
    localparam int ADDR_WIDTH = 9;
    localparam int kernel_size = 3;
    localparam int act_size = 5;
    localparam int W_ADDR = 0;
    localparam int A_ADDR = 100;
    localparam int P_ADDR = 500;

    logic clk = 0, reset = 1;
    logic [DATA_WIDTH-1:0] act_in = 0, filt_in = 0, pe_out;
    logic load_en_wght = 0, load_en_act = 0, start = 0;
    logic compute_done, load_done;

    PE #(
        .DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH),
        .W_READ_ADDR(W_ADDR), .A_READ_ADDR(A_ADDR),
        .W_LOAD_ADDR(W_ADDR), .A_LOAD_ADDR(A_ADDR),
        .PSUM_ADDR(P_ADDR), .kernel_size(kernel_size), .act_size(act_size)
    ) dut (
        .clk(clk), .reset(reset), .act_in(act_in), .filt_in(filt_in),
        .load_en_wght(load_en_wght), .load_en_act(load_en_act),
        .start(start), .pe_out(pe_out), .compute_done(compute_done), .load_done(load_done)
    );

    always #5 clk = ~clk; // 10ns period (posedge at 5, 15, 25, ...)

    int cyc = 0;
    always @(posedge clk) if (reset) cyc <= 0; else cyc <= cyc + 1;

    // Trace internal state transitions and SPad reads during compute
    always @(posedge clk) begin
        if (!reset && dut.state != 3'b000 && dut.state != 3'b101 && dut.state != 3'b110) begin
            $display("  [cyc=%0d] state=%0d filt_cnt=%0d iter=%0d r_addr=%0d r_data=%0d filt_reg=%0d act_reg=%0d psum_reg=%0d done=%0b",
                cyc, dut.state, dut.filt_count, dut.iter, dut.r_addr, dut.r_data,
                dut.filt_in_reg, dut.act_in_reg, dut.psum_reg, compute_done);
        end
    end

    function automatic int get_w(input int idx);
        case (idx)
            0: return 10; 1: return 20; 2: return 30;
            3: return 1;  4: return 2;  5: return 3;
            6: return 4;  7: return 5;  default: return 6;
        endcase
    endfunction

    function automatic int get_a(input int idx);
        if (idx < 15) return idx + 2;
        else if (idx < 20) return 1;
        else return 2;
    endfunction

    task automatic run_test(input bit corrupt_w1);
        int c0, c_load_w, c_load_a, c_iter0, c_iter1, c_iter2;
        int got0, got1, got2;
        int iter_before;
        @(negedge clk); reset = 1;
        @(negedge clk); @(negedge clk); reset = 0;

        // 1. Load 9 weights
        c0 = cyc;
        for (int i = 0; i < 9; i++) begin
            load_en_wght = (i == 0) ? 1'b1 : 1'b0;
            filt_in = (corrupt_w1 && i == 1) ? 16'd21 : 16'(get_w(i));
            @(negedge clk);
        end
        load_en_wght = 0;
        while (!load_done) @(negedge clk);
        c_load_w = cyc - c0;
        @(negedge clk); // return to IDLE

        // 2. Load 25 activations
        c0 = cyc;
        for (int i = 0; i < 25; i++) begin
            load_en_act = (i == 0) ? 1'b1 : 1'b0;
            act_in = 16'(get_a(i));
            @(negedge clk);
        end
        load_en_act = 0;
        while (!load_done) @(negedge clk);
        c_load_a = cyc - c0;
        @(negedge clk); // return to IDLE

        $display("=== PE RUN (corrupt_w1=%0b): LOAD_W_cycles=%0d LOAD_A_cycles=%0d ===", corrupt_w1, c_load_w, c_load_a);

        // 3. Run 3 valid iterations with strict 1-cycle start pulse
        for (int it = 0; it < 3; it++) begin
            int c_it, g_it;
            c0 = cyc;
            start = 1; @(negedge clk); start = 0;
            while (!compute_done) @(negedge clk);
            c_it = cyc - c0;
            g_it = int'(pe_out);
            @(negedge clk); // let WRITE finish and write SPad[500+it]
            $display("ITER %0d: pe_out=%0d spad_psum[%0d]=%0d cycles=%0d",
                it, g_it, P_ADDR + it, dut.spad_pe0.mem[P_ADDR + it], c_it);
        end

        // 4. Test 4th 1-cycle start pulse (when iter == 3 == act_size - kernel_size + 1)
        if (!corrupt_w1) begin
            iter_before = int'(dut.iter);
            start = 1; @(negedge clk); start = 0;
            $display("PULSE_4 (1-cycle): iter_before=%0d -> iter_after=%0d state=%0d compute_done=%0b",
                iter_before, dut.iter, dut.state, compute_done);
            // 5th 1-cycle start pulse should now run iter=0 again
            c0 = cyc;
            start = 1; @(negedge clk); start = 0;
            while (!compute_done) @(negedge clk);
            $display("PULSE_5 (after wrap): pe_out=%0d cycles=%0d", pe_out, cyc - c0);
        end
    endtask

    initial begin
        run_test(1'b0);
        run_test(1'b1);
        $finish;
    end
endmodule
