`timescale 1ns / 1ps
// E4 (Eyeriss v2 study): does the repository's PE.sv do anything different when weights/activations are zero?
// Drives the same unmodified PE as tb_e2_pe_row.sv with five data patterns and counts, per 1-D row iteration:
//   cycles from the 1-cycle start pulse to compute_done, posedges with mac_en=1, posedges with read_en=1 in READ_W/READ_A.
// It also prints how many (weight, activation) pairs were both non-zero, which is what a zero-skipping PE would have to process.
module tb_e4_pe_zero();
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

    always #5 clk = ~clk;

    int cyc = 0;
    always @(posedge clk) if (reset) cyc <= 0; else cyc <= cyc + 1;

    int cnt_mac, cnt_read;
    always @(posedge clk) begin
        if (!reset) begin
            if (dut.mac_en) cnt_mac <= cnt_mac + 1;
            if (dut.read_en && (dut.state == 3'b001 || dut.state == 3'b010)) cnt_read <= cnt_read + 1;
        end
    end

    int w_arr [0:8];
    int a_arr [0:24];

    task automatic fill(input int wmode, input int amode);
        for (int i = 0; i < 9; i++) begin
            case (wmode)
                0: w_arr[i] = (i == 0) ? 10 : (i == 1) ? 20 : (i == 2) ? 30 : i;        // dense
                1: w_arr[i] = (i == 1) ? 20 : 0;                                          // 2 of first 3 weights zero
                default: w_arr[i] = 0;                                                    // all zero
            endcase
        end
        for (int i = 0; i < 25; i++) begin
            case (amode)
                0: a_arr[i] = i + 2;                                                      // dense
                1: a_arr[i] = (i % 2 == 0) ? i + 2 : 0;                                   // every other activation zero
                default: a_arr[i] = 0;                                                    // all zero
            endcase
        end
    endtask

    task automatic run_case(input string name, input int wmode, input int amode);
        int c0, ph;
        fill(wmode, amode);
        cnt_mac = 0; cnt_read = 0;
        @(negedge clk); reset = 1;
        @(negedge clk); @(negedge clk); reset = 0;

        for (int i = 0; i < 9; i++) begin
            load_en_wght = (i == 0); filt_in = 16'(w_arr[i]); @(negedge clk);
        end
        load_en_wght = 0; while (!load_done) @(negedge clk); @(negedge clk);
        for (int i = 0; i < 25; i++) begin
            load_en_act = (i == 0); act_in = 16'(a_arr[i]); @(negedge clk);
        end
        load_en_act = 0; while (!load_done) @(negedge clk); @(negedge clk);

        for (int it = 0; it < 3; it++) begin
            int exp_v, nz, m0, r0;
            exp_v = 0; nz = 0;
            for (int j = 0; j < 3; j++) begin
                exp_v += w_arr[j] * a_arr[it*act_size + j];
                if (w_arr[j] != 0 && a_arr[it*act_size + j] != 0) nz++;
            end
            m0 = cnt_mac; r0 = cnt_read; c0 = cyc;
            start = 1; @(negedge clk); start = 0;
            while (!compute_done) @(negedge clk);
            $display("CASE %-24s ITER %0d: cycles=%0d mac_en=%0d spad_reads=%0d pe_out=%0d expected=%0d nonzero_pairs=%0d",
                     name, it, cyc - c0, cnt_mac - m0, cnt_read - r0, int'(pe_out), exp_v & 16'hFFFF, nz);
            @(negedge clk);
        end
    endtask

    initial begin
        run_case("dense", 0, 0);
        run_case("w=[0,20,0]", 1, 0);
        run_case("w=[0,0,0]", 2, 0);
        run_case("acts: every other zero", 0, 1);
        run_case("acts: all zero", 0, 2);
        run_case("w=[0,20,0] acts alt zero", 1, 1);
        $finish;
    end
endmodule
