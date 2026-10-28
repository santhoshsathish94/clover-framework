`timescale 1ns / 1ns

// Output-Stationary (OS) 2D/3D Convolution Testbench on an E x F grid of unmodified src/PE.v instances
// Each PE[x][y] holds output pixel(s) O[m][x][y] stationary in psum_spad.regs[m].
module tb_os_conv2d_e4;
    parameter integer R = 3;
    parameter integer H = 5;
    parameter integer W = 5;
    parameter integer C_IN = 1;
    parameter integer M_OUT = 1;
    parameter integer SPARSE = 0;
    parameter integer CORRUPT_WEIGHT = 0;

    localparam integer E = H - R + 1;
    localparam integer F = W - R + 1;

    reg clk, rstn, en;
    always #5 clk = ~clk;

    reg [15:0] I_mat   [0:C_IN-1][0:H-1][0:W-1];
    reg [15:0] FLT_mat [0:M_OUT-1][0:C_IN-1][0:R-1][0:R-1];

    reg  iact_wr   [0:E-1][0:F-1];
    reg  weight_wr [0:E-1][0:F-1];
    reg  [15:0] iact_in   [0:E-1][0:F-1];
    reg  [15:0] weight_in [0:E-1][0:F-1];
    wire [15:0] psum_out  [0:E-1][0:F-1];
    wire iact_rdy  [0:E-1][0:F-1];
    wire weight_rdy[0:E-1][0:F-1];
    wire psum_rdy  [0:E-1][0:F-1];
    wire psum_vld  [0:E-1][0:F-1];

    reg  override_rd_addr;
    reg  [5:0] drv_ifmap_raddr;
    reg  [7:0] drv_filter_raddr;
    reg  override_paddr;
    reg  [4:0] cur_paddr, paddr_q1, paddr_q2, drv_paddr;
    reg  freeze_spad_writes, hold_pipe_until_loaded, force_drain_fifo;
    wire [15:0] pe_psum_reg [0:E-1][0:F-1][0:M_OUT-1];

    always @(posedge clk) begin
        if (!rstn) begin
            paddr_q1 <= 0; paddr_q2 <= 0; drv_paddr <= 0;
        end else begin
            paddr_q1 <= cur_paddr; paddr_q2 <= paddr_q1; drv_paddr <= paddr_q2;
        end
    end

    integer cnt_ifmap_spad_wr, cnt_filter_spad_wr;
    integer cnt_ifmap_spad_rd, cnt_filter_spad_rd;
    integer cnt_psum_spad_rd, cnt_psum_spad_wr;
    integer cnt_mac_active, cnt_mac_nonzero, cnt_mac_zero_gated;
    integer cnt_inter_pe_psum_add;
    reg count_compute_window, is_padding_slot;

    genvar gx, gy, gm;
    generate
        for (gx = 0; gx < E; gx = gx + 1) begin : pe_x
            for (gy = 0; gy < F; gy = gy + 1) begin : pe_y
                PE #(
                    .IACT_SPAD_DEPTH(64),
                    .WEIGHT_SPAD_DEPTH(224),
                    .PSUM_SPAD_DEPTH(32),
                    .IACT_BUFFER_DEPTH(64),
                    .WEIGHT_BUFFER_DEPTH(64),
                    .WEIGHT_PAR_WRITE(1),
                    .PSUM_PAR_WRITE(1),
                    .PSUM_PAR_READ(1)
                ) dut (
                    .clk(clk), .rstn(rstn), .en(en),
                    .iact_write_en(iact_wr[gx][gy]),
                    .weight_write_en(weight_wr[gx][gy]),
                    .psum_write_en(1'b0), .psum_read_en(1'b0),
                    .data_iact_in(iact_in[gx][gy]),
                    .data_weight_in(weight_in[gx][gy]),
                    .data_psum_in(16'd0),
                    .filter_size(8'd4), .stride(8'd1),
                    .input_channels_num(8'd1), .output_channels_num(8'd1),
                    .data_psum_out(psum_out[gx][gy]),
                    .iact_buffer_ready(iact_rdy[gx][gy]),
                    .weight_buffer_ready(weight_rdy[gx][gy]),
                    .psum_buffer_ready(psum_rdy[gx][gy]),
                    .psum_out_valid(psum_vld[gx][gy])
                );

                for (gm = 0; gm < M_OUT; gm = gm + 1) begin : psum_tap
                    assign pe_psum_reg[gx][gy][gm] = dut.psum_spad.regs[gm];
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
                        if (dut.psum_spad.write_en && (drv_paddr < M_OUT) && !hold_pipe_until_loaded) begin
                            cnt_psum_spad_wr = cnt_psum_spad_wr + 1;
                            cnt_psum_spad_rd = cnt_psum_spad_rd + 1;
                        end
                    end
                end
            end
        end
    endgenerate

    integer r, s, ch, m, h_i, w_i, x, y;

    initial begin
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
            I_mat[0][1][2] = 16'd0; I_mat[0][3][0] = 16'd0; FLT_mat[0][0][0][2] = 16'd0;
        end
        if (CORRUPT_WEIGHT) FLT_mat[0][0][1][1] = 16'd99;

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

        for (x = 0; x < E; x = x + 1)
            for (y = 0; y < F; y = y + 1) begin
                iact_wr[x][y] = 0; weight_wr[x][y] = 0; iact_in[x][y] = 0; weight_in[x][y] = 0;
            end

        #10; rstn = 1; en = 1; #10;

        // 1. Stream receptive-field ifmap values I[ch][x+r][y+s] (C_IN * R * R words) into each PE[x][y]
        for (ch = 0; ch < C_IN; ch = ch + 1)
            for (r = 0; r < R; r = r + 1)
                for (s = 0; s < R; s = s + 1) begin
                    @(negedge clk);
                    for (x = 0; x < E; x = x + 1)
                        for (y = 0; y < F; y = y + 1) begin
                            iact_wr[x][y] = 1'b1;
                            iact_in[x][y] = I_mat[ch][x + r][y + s];
                        end
                    @(posedge clk);
                end
        @(negedge clk);
        for (x = 0; x < E; x = x + 1)
            for (y = 0; y < F; y = y + 1) iact_wr[x][y] = 1'b0;
        @(posedge clk);

        // 2. Broadcast all M_OUT * C_IN * R * R filter weights into each PE[x][y]
        for (m = 0; m < M_OUT; m = m + 1)
            for (ch = 0; ch < C_IN; ch = ch + 1)
                for (r = 0; r < R; r = r + 1)
                    for (s = 0; s < R; s = s + 1) begin
                        @(negedge clk);
                        for (x = 0; x < E; x = x + 1)
                            for (y = 0; y < F; y = y + 1) begin
                                weight_wr[x][y] = 1'b1;
                                weight_in[x][y] = FLT_mat[m][ch][r][s];
                            end
                        @(posedge clk);
                    end
        @(negedge clk);
        for (x = 0; x < E; x = x + 1)
            for (y = 0; y < F; y = y + 1) weight_wr[x][y] = 1'b0;
        @(posedge clk);

        // Zero guard slot at filter_spad[M_OUT*C_IN*R*R]
        @(negedge clk);
        is_padding_slot = 1'b1;
        for (x = 0; x < E; x = x + 1)
            for (y = 0; y < F; y = y + 1) begin
                weight_wr[x][y] = 1'b1; weight_in[x][y] = 16'd0;
            end
        @(posedge clk);
        @(negedge clk);
        for (x = 0; x < E; x = x + 1)
            for (y = 0; y < F; y = y + 1) weight_wr[x][y] = 1'b0;
        @(posedge clk);

        // 3. Compute stationary output pixels O[m][x][y] inside PE[x][y].psum_spad.regs[m]
        override_rd_addr = 1'b1;
        for (m = 0; m < M_OUT; m = m + 1)
            for (ch = 0; ch < C_IN; ch = ch + 1)
                for (r = 0; r < R; r = r + 1)
                    for (s = 0; s < R; s = s + 1) begin
                        @(negedge clk);
                        force_drain_fifo = 1'b0;
                        freeze_spad_writes = 1'b1;
                        hold_pipe_until_loaded = 1'b0;
                        count_compute_window = 1'b1;
                        is_padding_slot = 1'b0;
                        drv_ifmap_raddr  = (ch * R + r) * R + s;
                        drv_filter_raddr = ((m * C_IN + ch) * R + r) * R + s;
                        cur_paddr        = m;
                        @(posedge clk);
                    end

        @(negedge clk);
        count_compute_window = 1'b0; is_padding_slot = 1'b1;
        drv_filter_raddr = M_OUT * C_IN * R * R; drv_ifmap_raddr = 0; cur_paddr = 5'd30;
        repeat (6) @(posedge clk);

        // In OS, each PE[x][y] already holds O[m][x][y] in psum_spad.regs[m] (0 inter-PE adds!)
        $display("CFG R=%0d H=%0d W=%0d C=%0d M=%0d E=%0d F=%0d PEs=%0d SPARSE=%0d CORRUPT=%0d",
            R, H, W, C_IN, M_OUT, E, F, E*F, SPARSE, CORRUPT_WEIGHT);
        $display("COUNTS macs=%0d mac_nonzero=%0d mac_gated=%0d w_wr=%0d i_wr=%0d w_rd=%0d i_rd=%0d psum_rd=%0d psum_wr=%0d vert_adds=%0d",
            cnt_mac_active, cnt_mac_nonzero, cnt_mac_zero_gated,
            cnt_filter_spad_wr, cnt_ifmap_spad_wr, cnt_filter_spad_rd, cnt_ifmap_spad_rd,
            cnt_psum_spad_rd, cnt_psum_spad_wr, cnt_inter_pe_psum_add);
        for (m = 0; m < M_OUT; m = m + 1)
            for (x = 0; x < E; x = x + 1)
                for (y = 0; y < F; y = y + 1)
                    $display("OUT m=%0d x=%0d y=%0d val=%0d", m, x, y, pe_psum_reg[x][y][m]);
        $finish;
    end
endmodule
