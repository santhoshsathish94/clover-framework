`timescale 1ns / 1ns

// Parameterized Row-Stationary 2D/3D Convolution Testbench on an R x E grid of unmodified src/PE.v instances
module tb_rs_conv2d_e3;
    parameter integer R = 3;
    parameter integer H = 5;
    parameter integer W = 5;
    parameter integer C_IN = 1;
    parameter integer M_OUT = 1;
    parameter integer SPARSE = 0;
    parameter integer CORRUPT_WEIGHT = 0;

    localparam integer E = H - R + 1;
    localparam integer F = W - R + 1;
    localparam integer TOTAL_MACS = M_OUT * C_IN * E * F * R * R;

    reg clk, rstn, en;
    always #5 clk = ~clk;

    // Input feature map I[ch][h][w] and Filter weights FLT[m][ch][r][s]
    reg [15:0] I_mat   [0:C_IN-1][0:H-1][0:W-1];
    reg [15:0] FLT_mat [0:M_OUT-1][0:C_IN-1][0:R-1][0:R-1];

    // Per-PE signals for R x E PE array
    reg  iact_wr   [0:R-1][0:E-1];
    reg  weight_wr [0:R-1][0:E-1];
    reg  [15:0] iact_in   [0:R-1][0:E-1];
    reg  [15:0] weight_in [0:R-1][0:E-1];
    wire [15:0] psum_out  [0:R-1][0:E-1];
    wire iact_rdy  [0:R-1][0:E-1];
    wire weight_rdy[0:R-1][0:E-1];
    wire psum_rdy  [0:R-1][0:E-1];
    wire psum_vld  [0:R-1][0:E-1];

    reg  override_rd_addr;
    reg  [4:0] drv_ifmap_raddr;
    reg  [7:0] drv_filter_raddr;
    reg  override_paddr;
    reg  [4:0] cur_paddr, paddr_q1, paddr_q2, drv_paddr;
    reg  freeze_spad_writes;
    reg  hold_pipe_until_loaded;
    reg  force_drain_fifo;
    wire [15:0] pe_psum_reg [0:R-1][0:E-1][0:M_OUT*F-1];

    always @(posedge clk) begin
        if (!rstn) begin
            paddr_q1  <= 0;
            paddr_q2  <= 0;
            drv_paddr <= 0;
        end else begin
            paddr_q1  <= cur_paddr;
            paddr_q2  <= paddr_q1;
            drv_paddr <= paddr_q2;
        end
    end

    // Event counters across the R x E PE array
    integer cnt_ifmap_spad_wr, cnt_filter_spad_wr;
    integer cnt_ifmap_spad_rd, cnt_filter_spad_rd;
    integer cnt_psum_spad_rd, cnt_psum_spad_wr;
    integer cnt_mac_active, cnt_mac_nonzero, cnt_mac_zero_gated;
    integer cnt_inter_pe_psum_add;
    reg count_compute_window, is_padding_slot;
    reg count_acc_window;

    always @(posedge clk) begin
        if (!rstn) begin
            count_acc_window <= 0;
        end else begin
            // Accumulation at psum_spad occurs 3 clock cycles after spad read
            count_acc_window <= count_compute_window && !is_padding_slot;
        end
    end

    genvar gr, gc, gk;
    generate
        for (gr = 0; gr < R; gr = gr + 1) begin : pe_row
            for (gc = 0; gc < E; gc = gc + 1) begin : pe_col
                PE #(
                    .IACT_SPAD_DEPTH(32),
                    .WEIGHT_SPAD_DEPTH(224),
                    .PSUM_SPAD_DEPTH(32),
                    .IACT_BUFFER_DEPTH(32),
                    .WEIGHT_BUFFER_DEPTH(32),
                    .WEIGHT_PAR_WRITE(1),
                    .PSUM_PAR_WRITE(1),
                    .PSUM_PAR_READ(1)
                ) dut (
                    .clk(clk), .rstn(rstn), .en(en),
                    .iact_write_en(iact_wr[gr][gc]),
                    .weight_write_en(weight_wr[gr][gc]),
                    .psum_write_en(1'b0), .psum_read_en(1'b0),
                    .data_iact_in(iact_in[gr][gc]),
                    .data_weight_in(weight_in[gr][gc]),
                    .data_psum_in(16'd0),
                    .filter_size(8'd4), .stride(8'd1),
                    .input_channels_num(8'd1), .output_channels_num(8'd1),
                    .data_psum_out(psum_out[gr][gc]),
                    .iact_buffer_ready(iact_rdy[gr][gc]),
                    .weight_buffer_ready(weight_rdy[gr][gc]),
                    .psum_buffer_ready(psum_rdy[gr][gc]),
                    .psum_out_valid(psum_vld[gr][gc])
                );

                for (gk = 0; gk < M_OUT * F; gk = gk + 1) begin : psum_tap
                    assign pe_psum_reg[gr][gc][gk] = dut.psum_spad.regs[gk];
                end

                always @(*) begin
                    if (force_drain_fifo) begin
                        force dut.iact_buffer_valid = !dut.iact_buffer_empty;
                        force dut.weight_buffer_valid = !dut.weight_buffer_empty;
                        force dut.iact_counter_en = !dut.iact_buffer_empty;
                        force dut.weight_counter_en = !dut.weight_buffer_empty;
                        force dut.iact_spad_controller_clear = 1'b0;
                        force dut.weight_spad_controller_clear = 1'b0;
                    end else begin
                        release dut.iact_buffer_valid;
                        release dut.weight_buffer_valid;
                        release dut.iact_spad_controller_clear;
                        release dut.weight_spad_controller_clear;
                        if (freeze_spad_writes) begin
                            force dut.iact_counter_en = 1'b0;
                            force dut.weight_counter_en = 1'b0;
                        end else begin
                            release dut.iact_counter_en;
                            release dut.weight_counter_en;
                        end
                    end
                    if (hold_pipe_until_loaded) force dut.pipe_en = 1'b0;
                    else                        release dut.pipe_en;
                    if (override_rd_addr) begin
                        force dut.ifmap_read_cout = drv_ifmap_raddr;
                        force dut.filter_read_cout = drv_filter_raddr;
                    end else begin
                        release dut.ifmap_read_cout;
                        release dut.filter_read_cout;
                    end
                    if (override_paddr) force dut.psum_write_addr_fr = drv_paddr;
                    else                release dut.psum_write_addr_fr;
                end

                always @(posedge clk) begin
                    if (rstn && en) begin
                        if (dut.ifmap_spad.write_en) cnt_ifmap_spad_wr = cnt_ifmap_spad_wr + 1;
                        if (dut.filter_spad.write_en && dut.filter_spad.chip_en && !is_padding_slot)
                            cnt_filter_spad_wr = cnt_filter_spad_wr + 1;
                        if (count_compute_window && !is_padding_slot && dut.pipe_en) begin
                            cnt_ifmap_spad_rd = cnt_ifmap_spad_rd + 1;
                            cnt_filter_spad_rd = cnt_filter_spad_rd + 1;
                            cnt_mac_active = cnt_mac_active + 1;
                            if (dut.zero_flag) cnt_mac_nonzero = cnt_mac_nonzero + 1;
                            else               cnt_mac_zero_gated = cnt_mac_zero_gated + 1;
                        end
                        if ( dut.psum_spad.write_en && (drv_paddr < M_OUT * F) && !hold_pipe_until_loaded ) begin
                            cnt_psum_spad_wr = cnt_psum_spad_wr + 1;
                            cnt_psum_spad_rd = cnt_psum_spad_rd + 1;
                        end
                    end
                end
            end
        end
    endgenerate

    integer r, c, ch, m, h_i, w_i, s, y;
    integer ofmap [0:M_OUT-1][0:E-1][0:F-1];
    integer col_sum;

    initial begin
        // Deterministic initialization matching verify_e3_golden.py
        for (ch = 0; ch < C_IN; ch = ch + 1)
            for (h_i = 0; h_i < H; h_i = h_i + 1)
                for (w_i = 0; w_i < W; w_i = w_i + 1)
                    I_mat[ch][h_i][w_i] = ((h_i * 3 + w_i * 2 + ch + 1) % 5) + 1;

        for (m = 0; m < M_OUT; m = m + 1)
            for (ch = 0; ch < C_IN; ch = ch + 1)
                for (r = 0; r < R; r = r + 1)
                    for (s = 0; s < R; s = s + 1)
                        FLT_mat[m][ch][r][s] = ((r * 2 + s + m * 3 + ch * 2 + 1) % 4) + 1;

        if (SPARSE) begin
            I_mat[0][1][2] = 16'd0;
            I_mat[0][3][0] = 16'd0;
            FLT_mat[0][0][0][2] = 16'd0;
        end
        if (CORRUPT_WEIGHT) begin
            FLT_mat[0][0][1][1] = 16'd99;
        end

        cnt_ifmap_spad_wr = 0; cnt_filter_spad_wr = 0;
        cnt_ifmap_spad_rd = 0; cnt_filter_spad_rd = 0;
        cnt_psum_spad_rd = 0;  cnt_psum_spad_wr = 0;
        cnt_mac_active = 0;    cnt_mac_nonzero = 0; cnt_mac_zero_gated = 0;
        cnt_inter_pe_psum_add = 0;
        clk = 0; rstn = 0; en = 0;
        override_rd_addr = 0; drv_ifmap_raddr = 0; drv_filter_raddr = 0;
        override_paddr = 1; cur_paddr = 5'd30;
        freeze_spad_writes = 0; hold_pipe_until_loaded = 1; force_drain_fifo = 1;
        count_compute_window = 0; is_padding_slot = 0;

        for (r = 0; r < R; r = r + 1)
            for (c = 0; c < E; c = c + 1) begin
                iact_wr[r][c] = 0; weight_wr[r][c] = 0;
                iact_in[r][c] = 0; weight_in[r][c] = 0;
            end

        #10; rstn = 1; en = 1; #10;

        // 1. Stream C_IN ifmap rows of length W diagonally to PE[r][c] (row r+c)
        for (ch = 0; ch < C_IN; ch = ch + 1) begin
            for (w_i = 0; w_i < W; w_i = w_i + 1) begin
                @(negedge clk);
                for (r = 0; r < R; r = r + 1)
                    for (c = 0; c < E; c = c + 1) begin
                        iact_wr[r][c] = 1'b1;
                        iact_in[r][c] = I_mat[ch][r + c][w_i];
                    end
                @(posedge clk);
            end
        end
        @(negedge clk);
        for (r = 0; r < R; r = r + 1)
            for (c = 0; c < E; c = c + 1) iact_wr[r][c] = 1'b0;
        @(posedge clk);

        // 2. Stream M_OUT * C_IN filter rows of length R horizontally to PE[r][0..E-1]
        for (m = 0; m < M_OUT; m = m + 1) begin
            for (ch = 0; ch < C_IN; ch = ch + 1) begin
                for (s = 0; s < R; s = s + 1) begin
                    @(negedge clk);
                    for (r = 0; r < R; r = r + 1)
                        for (c = 0; c < E; c = c + 1) begin
                            weight_wr[r][c] = 1'b1;
                            weight_in[r][c] = FLT_mat[m][ch][r][s];
                        end
                    @(posedge clk);
                end
            end
        end
        // Drain the final real weight from weight_fifo into filter_spad before setting is_padding_slot
        @(negedge clk);
        for (r = 0; r < R; r = r + 1)
            for (c = 0; c < E; c = c + 1) weight_wr[r][c] = 1'b0;
        @(posedge clk);

        // Write 1 zero guard slot at filter_spad[M_OUT*C_IN*R] for pipeline drain
        @(negedge clk);
        is_padding_slot = 1'b1;
        for (r = 0; r < R; r = r + 1)
            for (c = 0; c < E; c = c + 1) begin
                weight_wr[r][c] = 1'b1; weight_in[r][c] = 16'd0;
            end
        @(posedge clk);
        @(negedge clk);
        for (r = 0; r < R; r = r + 1)
            for (c = 0; c < E; c = c + 1) weight_wr[r][c] = 1'b0;
        @(posedge clk);

        // 3. Execute 1D convolution primitives across all M_OUT, C_IN, y=0..F-1, s=0..R-1
        override_rd_addr = 1'b1;
        for (m = 0; m < M_OUT; m = m + 1) begin
            for (ch = 0; ch < C_IN; ch = ch + 1) begin
                for (y = 0; y < F; y = y + 1) begin
                    for (s = 0; s < R; s = s + 1) begin
                        @(negedge clk);
                        force_drain_fifo = 1'b0;
                        freeze_spad_writes = 1'b1;
                        hold_pipe_until_loaded = 1'b0;
                        count_compute_window = 1'b1;
                        is_padding_slot = 1'b0;
                        drv_filter_raddr = (m * C_IN + ch) * R + s;
                        drv_ifmap_raddr  = ch * W + y + s;
                        cur_paddr        = m * F + y;
                        @(posedge clk);
                    end
                end
            end
        end

        // Drain pipeline
        @(negedge clk);
        count_compute_window = 1'b0;
        is_padding_slot = 1'b1;
        drv_filter_raddr = M_OUT * C_IN * R;
        drv_ifmap_raddr  = 0;
        cur_paddr        = 5'd30;
        repeat (6) @(posedge clk);

        // 4. Vertical accumulation across the R rows of PEs for each (m, c, y)
        for (m = 0; m < M_OUT; m = m + 1) begin
            for (c = 0; c < E; c = c + 1) begin
                for (y = 0; y < F; y = y + 1) begin
                    col_sum = 0;
                    for (r = 0; r < R; r = r + 1) begin
                        col_sum = col_sum + pe_psum_reg[r][c][m * F + y];
                    end
                    ofmap[m][c][y] = col_sum;
                    cnt_inter_pe_psum_add = cnt_inter_pe_psum_add + (R - 1);
                end
            end
        end

        $display("CFG R=%0d H=%0d W=%0d C=%0d M=%0d E=%0d F=%0d PEs=%0d SPARSE=%0d CORRUPT=%0d",
            R, H, W, C_IN, M_OUT, E, F, R*E, SPARSE, CORRUPT_WEIGHT);
        $display("COUNTS macs=%0d mac_nonzero=%0d mac_gated=%0d w_wr=%0d i_wr=%0d w_rd=%0d i_rd=%0d psum_rd=%0d psum_wr=%0d vert_adds=%0d",
            cnt_mac_active, cnt_mac_nonzero, cnt_mac_zero_gated,
            cnt_filter_spad_wr, cnt_ifmap_spad_wr, cnt_filter_spad_rd, cnt_ifmap_spad_rd,
            cnt_psum_spad_rd, cnt_psum_spad_wr, cnt_inter_pe_psum_add);
        for (m = 0; m < M_OUT; m = m + 1)
            for (c = 0; c < E; c = c + 1)
                for (y = 0; y < F; y = y + 1)
                    $display("OUT m=%0d x=%0d y=%0d val=%0d", m, c, y, ofmap[m][c][y]);
        $finish;
    end
endmodule
