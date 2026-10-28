`timescale 1ns/1ns

module tb_pe_e2;
    parameter MAX_CONFIG_WIDTH = 8;
    parameter DATA_WIDTH = 16;
    parameter IACT_SPAD_DEPTH = 12;
    parameter WEIGHT_SPAD_DEPTH = 224;
    parameter PSUM_SPAD_DEPTH = 24;
    parameter IACT_BUFFER_DEPTH = 16;
    parameter WEIGHT_BUFFER_DEPTH = 16;
    parameter WEIGHT_PAR_WRITE = 4;
    parameter PSUM_IN_BUFFER_DEPTH = 8;
    parameter PSUM_OUT_BUFFER_DEPTH = 8;

    reg clk, rstn, en;
    reg iact_write_en, weight_write_en, psum_write_en, psum_read_en;
    reg [DATA_WIDTH-1:0] data_iact_in;
    reg [WEIGHT_PAR_WRITE*DATA_WIDTH-1:0] data_weight_in;
    reg [DATA_WIDTH-1:0] data_psum_in;
    wire [DATA_WIDTH-1:0] data_psum_out;
    wire iact_buffer_ready, weight_buffer_ready, psum_buffer_ready, psum_out_valid;

    PE #(
        .MAX_CONFIG_WIDTH(MAX_CONFIG_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .IACT_SPAD_DEPTH(IACT_SPAD_DEPTH),
        .IACT_SPAD_WIDTH(DATA_WIDTH),
        .WEIGHT_SPAD_DEPTH(WEIGHT_SPAD_DEPTH),
        .WEIGHT_SPAD_WIDTH(DATA_WIDTH),
        .PSUM_SPAD_DEPTH(PSUM_SPAD_DEPTH),
        .PSUM_SPAD_WIDTH(DATA_WIDTH),
        .IACT_BUFFER_DEPTH(IACT_BUFFER_DEPTH),
        .IACT_BUFFER_WIDTH(DATA_WIDTH),
        .WEIGHT_BUFFER_DEPTH(WEIGHT_BUFFER_DEPTH),
        .WEIGHT_BUFFER_WIDTH(DATA_WIDTH),
        .WEIGHT_PAR_WRITE(WEIGHT_PAR_WRITE),
        .PSUM_IN_BUFFER_DEPTH(PSUM_IN_BUFFER_DEPTH),
        .PSUM_OUT_BUFFER_DEPTH(PSUM_OUT_BUFFER_DEPTH),
        .PSUM_BUFFER_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk), .rstn(rstn), .en(en),
        .iact_write_en(iact_write_en), .weight_write_en(weight_write_en),
        .psum_write_en(psum_write_en), .psum_read_en(psum_read_en),
        .data_iact_in(data_iact_in), .data_weight_in(data_weight_in),
        .data_psum_in(data_psum_in),
        .filter_size(8'd4), .stride(8'd1),
        .input_channels_num(8'd1), .output_channels_num(8'd1),
        .data_psum_out(data_psum_out),
        .iact_buffer_ready(iact_buffer_ready),
        .weight_buffer_ready(weight_buffer_ready),
        .psum_buffer_ready(psum_buffer_ready),
        .psum_out_valid(psum_out_valid)
    );

    always #5 clk = ~clk;

    // Hardware activity counters during the S=4 1D convolution window
    integer cnt_w_spad_wr, cnt_i_spad_wr;
    integer cnt_w_spad_rd, cnt_i_spad_rd;
    integer cnt_dff_latch, cnt_mult_en;
    integer cnt_psum_rd_acc, cnt_psum_wr_acc, cnt_psum_wr_all;
    reg count_spad_wr, count_window_read, count_window_acc;

    always @(posedge clk) begin
        if (rstn && en) begin
            if (count_spad_wr && dut.weight_counter_en) cnt_w_spad_wr = cnt_w_spad_wr + 1;
            if (count_spad_wr && dut.iact_counter_en)   cnt_i_spad_wr = cnt_i_spad_wr + 1;
            if (dut.psum_zero_flag)                     cnt_psum_wr_all = cnt_psum_wr_all + 1;
            if (count_window_read && dut.pipe_en) begin
                cnt_w_spad_rd = cnt_w_spad_rd + 1;
                cnt_i_spad_rd = cnt_i_spad_rd + 1;
                if (dut.zero_flag) cnt_dff_latch = cnt_dff_latch + 1;
            end
            if (count_window_acc && dut.pipe_en) begin
                cnt_mult_en = cnt_mult_en + 1;
                cnt_psum_rd_acc = cnt_psum_rd_acc + 1;
                if (dut.psum_zero_flag) cnt_psum_wr_acc = cnt_psum_wr_acc + 1;
            end
        end
    end

    // Drive 4 weights and 4 iacts keeping write_en high for 3 cycles so BufferController
    // stays in S_WRITE_BUFFER (valid=1) while all 4 words drain into filter_spad[0..3] and ifmap_spad[0..3]
    task load_4_weights_and_4_iacts(
        input [15:0] w0, input [15:0] w1, input [15:0] w2, input [15:0] w3,
        input [15:0] i0, input [15:0] i1, input [15:0] i2, input [15:0] i3
    );
        begin
            count_spad_wr = 1;
            @(negedge clk);
            weight_write_en = 1;
            data_weight_in = {w3, w2, w1, w0};
            iact_write_en = 1;
            data_iact_in = i0;
            @(negedge clk);
            data_weight_in = {16'd10, 16'd10, 16'd10, 16'd10}; // addresses 4..7 if read
            data_iact_in = i1;
            @(negedge clk);
            data_iact_in = i2;
            @(negedge clk);
            weight_write_en = 0;
            data_iact_in = i3;
            @(negedge clk);
            iact_write_en = 0;
        end
    endtask

    integer sel_case;
    initial begin
        if (!$value$plusargs("CASE=%d", sel_case)) sel_case = 1;
        clk = 0; rstn = 0; en = 0;
        iact_write_en = 0; weight_write_en = 0; psum_write_en = 0; psum_read_en = 0;
        data_iact_in = 0; data_weight_in = 0; data_psum_in = 0;
        cnt_w_spad_wr = 0; cnt_i_spad_wr = 0; cnt_w_spad_rd = 0; cnt_i_spad_rd = 0;
        cnt_dff_latch = 0; cnt_mult_en = 0; cnt_psum_rd_acc = 0; cnt_psum_wr_acc = 0; cnt_psum_wr_all = 0;
        count_spad_wr = 0; count_window_read = 0; count_window_acc = 0;
        #20; rstn = 1; en = 1;

        if (sel_case == 1) begin
            load_4_weights_and_4_iacts(16'd3, 16'd2, 16'd5, 16'd4, 16'd2, 16'd4, 16'd1, 16'd3);
            wait(dut.pipe_en == 1'b1);
            count_spad_wr = 0;
            count_window_read = 1;
            repeat(3) @(posedge clk);
            count_window_acc = 1;
            @(posedge clk);
            count_window_read = 0;
            repeat(3) @(posedge clk);
            count_window_acc = 0;
            #2;
            $display("[CASE 1: Dense 1D W=[3,2,5,4], I=[2,4,1,3]] psum_spad[0]=%0d (expected=31) | w_wr=%0d i_wr=%0d w_rd=%0d i_rd=%0d dff_latch=%0d mult_en=%0d psum_rd=%0d psum_wr_acc=%0d psum_wr_all=%0d",
                dut.psum_spad.regs[0], cnt_w_spad_wr, cnt_i_spad_wr, cnt_w_spad_rd, cnt_i_spad_rd,
                cnt_dff_latch, cnt_mult_en, cnt_psum_rd_acc, cnt_psum_wr_acc, cnt_psum_wr_all);
        end else if (sel_case == 2) begin
            load_4_weights_and_4_iacts(16'd9, 16'd2, 16'd5, 16'd4, 16'd2, 16'd4, 16'd1, 16'd3);
            wait(dut.pipe_en == 1'b1);
            repeat(7) @(posedge clk);
            #2;
            $display("[CASE 1b: Negative Control W[0]=9] psum_spad[0]=%0d (expected=43 != 31) | mismatch_detected=%b",
                dut.psum_spad.regs[0], (dut.psum_spad.regs[0] != 16'd31));
        end else if (sel_case == 3) begin
            load_4_weights_and_4_iacts(16'd3, 16'd2, 16'd0, 16'd4, 16'd2, 16'd4, 16'd1, 16'd3);
            wait(dut.pipe_en == 1'b1);
            count_spad_wr = 0;
            count_window_read = 1;
            repeat(3) @(posedge clk);
            count_window_acc = 1;
            @(posedge clk);
            count_window_read = 0;
            repeat(3) @(posedge clk);
            count_window_acc = 0;
            #2;
            $display("[CASE 2: Zero Weight W=[3,2,0,4], I=[2,4,1,3]] psum_spad[0]=%0d (expected=26) | w_wr=%0d i_wr=%0d w_rd=%0d i_rd=%0d dff_latch=%0d mult_en=%0d psum_rd=%0d psum_wr_acc=%0d psum_wr_all=%0d",
                dut.psum_spad.regs[0], cnt_w_spad_wr, cnt_i_spad_wr, cnt_w_spad_rd, cnt_i_spad_rd,
                cnt_dff_latch, cnt_mult_en, cnt_psum_rd_acc, cnt_psum_wr_acc, cnt_psum_wr_all);
        end else if (sel_case == 4) begin
            dut.filter_spad.mem[4] = 16'd10;
            count_spad_wr = 1;
            @(negedge clk);
            weight_write_en = 1; data_weight_in = {16'd4, 16'd5, 16'd2, 16'd3};
            iact_write_en = 1;   data_iact_in = 16'd2;
            @(negedge clk);
            data_weight_in = {16'd10, 16'd10, 16'd10, 16'd10};
            data_iact_in = 16'd4;
            @(negedge clk); data_iact_in = 16'd1;
            @(negedge clk); weight_write_en = 0; data_iact_in = 16'd3;
            @(negedge clk); iact_write_en = 1; data_iact_in = 16'd1;
            @(negedge clk); data_iact_in = 16'd5;
            @(negedge clk); data_iact_in = 16'd2;
            @(negedge clk); data_iact_in = 16'd3;
            @(negedge clk); iact_write_en = 0;
            repeat(10) @(posedge clk);
            #2;
            $display("[CASE 3: Back-to-back rows] psum_spad[0]=%0d (row0 expected=31), psum_spad[1]=%0d (unmod expected=110, wrapped expected=35) | final_wread=%0d",
                dut.psum_spad.regs[0], dut.psum_spad.regs[1], dut.filter_read_cout);
        end else if (sel_case == 5) begin
            load_4_weights_and_4_iacts(16'd3, 16'd2, 16'd5, 16'd4, 16'd2, 16'd4, 16'd1, 16'd3);
            wait(dut.pipe_en == 1'b1);
            repeat(5) @(posedge clk);
            #2;
            $display("[CASE 4a: Mid-run before reset] psum_spad[0]=%0d (expected=14), ifmap_spad[0]=%0d, filter_spad[0]=%0d, loc_en=%b, mul_valid=%b",
                dut.psum_spad.regs[0], dut.ifmap_spad.regs[0], dut.filter_spad.mem[0],
                dut.filter_ifmap_coordinator.loc_en, dut.multiplier.mul_valid);
            rstn = 0;
            @(posedge clk); #2;
            $display("[CASE 4b: During rstn=0] psum_spad[0]=%0d (expected=0), ifmap_spad[0]=%0d (expected=0), filter_spad[0]=%0d (SRAM retains=3), loc_en=%b, mul_valid=%b",
                dut.psum_spad.regs[0], dut.ifmap_spad.regs[0], dut.filter_spad.mem[0],
                dut.filter_ifmap_coordinator.loc_en, dut.multiplier.mul_valid);
            rstn = 1;
            load_4_weights_and_4_iacts(16'd3, 16'd2, 16'd5, 16'd4, 16'd2, 16'd4, 16'd1, 16'd3);
            wait(dut.pipe_en == 1'b1);
            repeat(7) @(posedge clk);
            #2;
            $display("[CASE 4c: After reset & reload] psum_spad[0..3]=[%0d, %0d, %0d, %0d] (expected psum_spad[0]=31 when reset is complete)",
                dut.psum_spad.regs[0], dut.psum_spad.regs[1], dut.psum_spad.regs[2], dut.psum_spad.regs[3]);
        end
        $finish;
    end
endmodule
