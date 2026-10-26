`timescale 1ns / 1ps
module tb_e3_scaling();
    localparam int DATA_WIDTH = 16;
    logic clk = 0, reset = 1;
    always #5 clk = ~clk;

    int cyc = 0;
    always @(posedge clk) if (reset) cyc <= 0; else cyc <= cyc + 1;

    // --- PART A: Problem-size scaling across (K,H) = (3,5), (3,6), (3,7), (5,7) ---
    logic [15:0] act_a = 0, filt_a = 0;
    logic lw_a = 0, la_a = 0, st_a = 0;
    logic [15:0] out_3x5[0:2], out_3x6[0:3], out_3x7[0:4], out_5x7[0:2];
    logic cd_3x5, ld_3x5, cd_3x6, ld_3x6, cd_3x7, ld_3x7, cd_5x7, ld_5x7;

    PE_cluster #(.X_dim(3), .Y_dim(3), .kernel_size(3), .act_size(5)) u_3x5 (
        .clk(clk), .reset(reset), .act_in(act_a), .filt_in(filt_a),
        .load_en_wght(lw_a), .load_en_act(la_a), .start(st_a),
        .pe_out(out_3x5), .compute_done(cd_3x5), .load_done(ld_3x5));

    PE_cluster #(.X_dim(4), .Y_dim(3), .kernel_size(3), .act_size(6)) u_3x6 (
        .clk(clk), .reset(reset), .act_in(act_a), .filt_in(filt_a),
        .load_en_wght(lw_a), .load_en_act(la_a), .start(st_a),
        .pe_out(out_3x6), .compute_done(cd_3x6), .load_done(ld_3x6));

    PE_cluster #(.X_dim(5), .Y_dim(3), .kernel_size(3), .act_size(7)) u_3x7 (
        .clk(clk), .reset(reset), .act_in(act_a), .filt_in(filt_a),
        .load_en_wght(lw_a), .load_en_act(la_a), .start(st_a),
        .pe_out(out_3x7), .compute_done(cd_3x7), .load_done(ld_3x7));

    PE_cluster #(.X_dim(3), .Y_dim(5), .kernel_size(5), .act_size(7)) u_5x7 (
        .clk(clk), .reset(reset), .act_in(act_a), .filt_in(filt_a),
        .load_en_wght(lw_a), .load_en_act(la_a), .start(st_a),
        .pe_out(out_5x7), .compute_done(cd_5x7), .load_done(ld_5x7));

    task automatic run_size_case(input int id, input int K, input int H);
        int E = H - K + 1;
        int c0, cw, ca, cc;
        @(negedge clk); reset = 1; @(negedge clk); @(negedge clk); reset = 0;
        c0 = cyc;
        for (int i = 1; i <= K*K; i++) begin
            lw_a = (i == 1); filt_a = 16'(i); @(negedge clk);
        end
        lw_a = 0; @(negedge clk); // load_done cycle
        cw = cyc - c0;

        c0 = cyc;
        for (int i = 1; i <= H*H; i++) begin
            la_a = (i == 1); act_a = 16'(i); @(negedge clk);
        end
        la_a = 0; @(negedge clk); // load_done cycle
        ca = cyc - c0;

        c0 = cyc;
        for (int r = 0; r < E; r++) begin
            st_a = 1; @(negedge clk); st_a = 0;
            if (id == 0) while (!cd_3x5) @(negedge clk);
            if (id == 1) while (!cd_3x6) @(negedge clk);
            if (id == 2) while (!cd_3x7) @(negedge clk);
            if (id == 3) while (!cd_5x7) @(negedge clk);
            @(negedge clk); // let psum_adder register pe_out
            if (id == 0) $display("SIZE(K=%0d,H=%0d) row=%0d out=[%0d,%0d,%0d]", K, H, r, out_3x5[0], out_3x5[1], out_3x5[2]);
            if (id == 1) $display("SIZE(K=%0d,H=%0d) row=%0d out=[%0d,%0d,%0d,%0d]", K, H, r, out_3x6[0], out_3x6[1], out_3x6[2], out_3x6[3]);
            if (id == 2) $display("SIZE(K=%0d,H=%0d) row=%0d out=[%0d,%0d,%0d,%0d,%0d]", K, H, r, out_3x7[0], out_3x7[1], out_3x7[2], out_3x7[3], out_3x7[4]);
            if (id == 3) $display("SIZE(K=%0d,H=%0d) row=%0d out=[%0d,%0d,%0d]", K, H, r, out_5x7[0], out_5x7[1], out_5x7[2]);
        end
        cc = cyc - c0;
        $display("SIZE_SUMMARY K=%0d H=%0d E=%0d PEs=%0d: load_w=%0d load_a=%0d compute=%0d total=%0d",
            K, H, E, E*K, cw, ca, cc, cw + ca + cc);
    endtask

    // --- PART B: 1, 2, 4 clusters running 4-channel 3x3x5x5 convolution ---
    // Initialize router_mode to 15 (default/disabled) so 2-state Verilator does not start all 4 routers in mode 0 (ALL)
    logic [3:0] mw_w0=15, mw_w1=15, mw_e0=15, mw_e1=15, ma_w0=15, ma_w1=15, ma_e0=15, ma_e1=15;
    logic [15:0] dw_i_w0=0, dw_i_w1=0, dw_i_e0=0, dw_i_e1=0;
    logic ew_i_w0=0, ew_i_w1=0, ew_i_e0=0, ew_i_e1=0;
    logic [15:0] dw_o_w0, dw_o_w1, dw_o_e0, dw_o_e1;
    logic ew_o_w0, ew_o_w1, ew_o_e0, ew_o_e1;

    logic [15:0] da_i_w0=0, da_i_w1=0, da_i_e0=0, da_i_e1=0;
    logic ea_i_w0=0, ea_i_w1=0, ea_i_e0=0, ea_i_e1=0;
    logic [15:0] da_o_w0, da_o_w1, da_o_e0, da_o_e1;
    logic ea_o_w0, ea_o_w1, ea_o_e0, ea_o_e1;

    router_network4 #(.DATA_WIDTH(16)) net_w (
        .router_mode_west_0(mw_w0), .west_data_i_west_0(dw_i_w0), .west_enable_i_west_0(ew_i_w0), .west_data_o_west_0(dw_o_w0), .west_enable_o_west_0(ew_o_w0),
        .router_mode_west_1(mw_w1), .west_data_i_west_1(dw_i_w1), .west_enable_i_west_1(ew_i_w1), .west_data_o_west_1(dw_o_w1), .west_enable_o_west_1(ew_o_w1),
        .router_mode_east_0(mw_e0), .east_data_i_east_0(dw_i_e0), .east_enable_i_east_0(ew_i_e0), .east_data_o_east_0(dw_o_e0), .east_enable_o_east_0(ew_o_e0),
        .router_mode_east_1(mw_e1), .east_data_i_east_1(dw_i_e1), .east_enable_i_east_1(ew_i_e1), .east_data_o_east_1(dw_o_e1), .east_enable_o_east_1(ew_o_e1)
    );
    router_network4 #(.DATA_WIDTH(16)) net_a (
        .router_mode_west_0(ma_w0), .west_data_i_west_0(da_i_w0), .west_enable_i_west_0(ea_i_w0), .west_data_o_west_0(da_o_w0), .west_enable_o_west_0(ea_o_w0),
        .router_mode_west_1(ma_w1), .west_data_i_west_1(da_i_w1), .west_enable_i_west_1(ea_i_w1), .west_data_o_west_1(da_o_w1), .west_enable_o_west_1(ea_o_w1),
        .router_mode_east_0(ma_e0), .east_data_i_east_0(da_i_e0), .east_enable_i_east_0(ea_i_e0), .east_data_o_east_0(da_o_e0), .east_enable_o_east_0(ea_o_e0),
        .router_mode_east_1(ma_e1), .east_data_i_east_1(da_i_e1), .east_enable_i_east_1(ea_i_e1), .east_data_o_east_1(da_o_e1), .east_enable_o_east_1(ea_o_e1)
    );

    logic lw_b=0, la_b=0, st_b=0, cd_b, ld_b;
    logic [15:0] po_w0[0:2], po_w1[0:2], po_e0[0:2], po_e1[0:2];

    PE_cluster #(.X_dim(3), .Y_dim(3), .kernel_size(3), .act_size(5)) c_w0 (
        .clk(clk), .reset(reset), .act_in(da_o_w0), .filt_in(dw_o_w0),
        .load_en_wght(lw_b), .load_en_act(la_b), .start(st_b), .pe_out(po_w0), .compute_done(cd_b), .load_done(ld_b));
    PE_cluster #(.X_dim(3), .Y_dim(3), .kernel_size(3), .act_size(5)) c_w1 (
        .clk(clk), .reset(reset), .act_in(da_o_w1), .filt_in(dw_o_w1),
        .load_en_wght(lw_b), .load_en_act(la_b), .start(st_b), .pe_out(po_w1), .compute_done(), .load_done());
    PE_cluster #(.X_dim(3), .Y_dim(3), .kernel_size(3), .act_size(5)) c_e0 (
        .clk(clk), .reset(reset), .act_in(da_o_e0), .filt_in(dw_o_e0),
        .load_en_wght(lw_b), .load_en_act(la_b), .start(st_b), .pe_out(po_e0), .compute_done(), .load_done());
    PE_cluster #(.X_dim(3), .Y_dim(3), .kernel_size(3), .act_size(5)) c_e1 (
        .clk(clk), .reset(reset), .act_in(da_o_e1), .filt_in(dw_o_e1),
        .load_en_wght(lw_b), .load_en_act(la_b), .start(st_b), .pe_out(po_e1), .compute_done(), .load_done());

    function automatic int ch_val(input int ch, input int i);
        case (ch)
            0: return i * 5;
            1: return i + 3;
            2: return i;
            default: return i + 2;
        endcase
    endfunction

    task automatic run_cluster_scaling(input int n_clusters, input bit corrupt_w4);
        int c0, passes = 4 / n_clusters;
        // Configure weight network: broadcast from west_0 (ALL=0, EASTSOUTH=6, WEST=3, EAST=4)
        mw_w0 = 4'd0; mw_e0 = 4'd6; mw_w1 = 4'd3; mw_e1 = 4'd4;
        // Configure iact network: unicast into each cluster (WEST=3, EAST=4)
        ma_w0 = 4'd3; ma_w1 = 4'd3; ma_e0 = 4'd4; ma_e1 = 4'd4;

        @(negedge clk); reset = 1; @(negedge clk); @(negedge clk); reset = 0;
        c0 = cyc;

        for (int p = 0; p < passes; p++) begin
            // Note: In PE.sv, after 3 compute iterations (r=0..2), iter==3 and filt_count==3,
            // and IDLE(load_en_act) does not reset filt_count<=0.
            // If p > 0, we wrap iter (1 cycle start=1 in IDLE) and reload weights (10 cycles) so filt_count resets to 0.
            if (p > 0) begin
                st_b = 1; @(negedge clk); st_b = 0; // wraps iter 3 -> 0 in 1 cycle
            end
            ew_i_w0 = 1;
            for (int i = 1; i <= 9; i++) begin
                lw_b = (i == 1);
                dw_i_w0 = (corrupt_w4 && i == 5) ? 16'd6 : 16'(i);
                @(negedge clk);
            end
            lw_b = 0; ew_i_w0 = 0; @(negedge clk);

            ea_i_w0 = 1; ea_i_e0 = (n_clusters >= 2); ea_i_w1 = (n_clusters == 4); ea_i_e1 = (n_clusters == 4);
            for (int i = 1; i <= 25; i++) begin
                la_b = (i == 1);
                if (n_clusters == 1) da_i_w0 = 16'(ch_val(p, i));
                else if (n_clusters == 2) begin
                    da_i_w0 = 16'(ch_val(2*p, i));
                    da_i_e0 = 16'(ch_val(2*p+1, i));
                end else begin
                    da_i_w0 = 16'(ch_val(0, i)); da_i_w1 = 16'(ch_val(1, i));
                    da_i_e0 = 16'(ch_val(2, i)); da_i_e1 = 16'(ch_val(3, i));
                end
                @(negedge clk);
            end
            la_b = 0; ea_i_w0 = 0; ea_i_e0 = 0; ea_i_w1 = 0; ea_i_e1 = 0; @(negedge clk);

            for (int r = 0; r < 3; r++) begin
                st_b = 1; @(negedge clk); st_b = 0;
                while (!cd_b) @(negedge clk);
                @(negedge clk);
                if (r == 2) begin
                    $display("CLUST(N=%0d,corrupt=%0b) pass=%0d row2: w0=[%0d,%0d,%0d] w1=[%0d,%0d,%0d] e0=[%0d,%0d,%0d] e1=[%0d,%0d,%0d]",
                        n_clusters, corrupt_w4, p,
                        po_w0[0], po_w0[1], po_w0[2], po_w1[0], po_w1[1], po_w1[2],
                        po_e0[0], po_e0[1], po_e0[2], po_e1[0], po_e1[1], po_e1[2]);
                end
            end
        end
        $display("CLUST_SUMMARY N=%0d corrupt=%0b passes=%0d active_cycles=%0d", n_clusters, corrupt_w4, passes, cyc - c0);
    endtask

    initial begin
        run_size_case(0, 3, 5);
        run_size_case(1, 3, 6);
        run_size_case(2, 3, 7);
        run_size_case(3, 5, 7);
        run_cluster_scaling(1, 1'b0);
        run_cluster_scaling(2, 1'b0);
        run_cluster_scaling(4, 1'b0);
        run_cluster_scaling(4, 1'b1);
        $finish;
    end
endmodule
