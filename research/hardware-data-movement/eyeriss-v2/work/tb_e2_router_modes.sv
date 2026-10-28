`timescale 1ns / 1ps
module tb_e2_router_modes();
    localparam int DATA_WIDTH = 16;

    logic [3:0] router_mode;
    logic [DATA_WIDTH-1:0] north_data_i, south_data_i, west_data_i, east_data_i;
    logic north_enable_i, south_enable_i, west_enable_i, east_enable_i;
    logic [DATA_WIDTH-1:0] north_data_o, south_data_o, west_data_o, east_data_o;
    logic north_enable_o, south_enable_o, west_enable_o, east_enable_o;

    router #(.DATA_WIDTH(DATA_WIDTH)) dut (
        .router_mode(router_mode),
        .north_data_i(north_data_i), .north_enable_i(north_enable_i),
        .north_data_o(north_data_o), .north_enable_o(north_enable_o),
        .south_data_i(south_data_i), .south_enable_i(south_enable_i),
        .south_data_o(south_data_o), .south_enable_o(south_enable_o),
        .west_data_i(west_data_i),   .west_enable_i(west_enable_i),
        .west_data_o(west_data_o),   .west_enable_o(west_enable_o),
        .east_data_i(east_data_i),   .east_enable_i(east_enable_i),
        .east_data_o(east_data_o),   .east_enable_o(east_enable_o)
    );

    string mode_names[16] = '{
        "0:ALL", "1:NORTH", "2:SOUTH", "3:WEST", "4:EAST",
        "5:EASTNORTH", "6:EASTSOUTH", "7:EASTWEST",
        "8:WESTNORTH", "9:WESTSOUTH", "10:WESTEAST",
        "11:DEF11", "12:DEF12", "13:DEF13", "14:DEF14", "15:DEF15"
    };

    string src_names[4] = '{"NORTH_IN(0x1111)", "SOUTH_IN(0x2222)", "WEST_IN(0x3333)", "EAST_IN(0x4444)"};
    int src_words[4]    = '{16'h1111, 16'h2222, 16'h3333, 16'h4444};

    initial begin
        north_data_i = 16'h1111;
        south_data_i = 16'h2222;
        west_data_i  = 16'h3333;
        east_data_i  = 16'h4444;

        $display("=== 1. SWEEP ALL 16 ROUTER MODES x 4 SINGLE-SOURCE INPUTS (0-CYCLE COMBINATIONAL) ===");
        for (int m = 0; m < 16; m++) begin
            router_mode = 4'(m);
            for (int s = 0; s < 4; s++) begin
                north_enable_i = (s == 0);
                south_enable_i = (s == 1);
                west_enable_i  = (s == 2);
                east_enable_i  = (s == 3);
                #1; // settle combinational delta
                if (s == 0 || m <= 10) begin
                    // Print full row for s==0 on all modes, plus verify all 4 sources route identically
                    if (s == 0) begin
                        $display("MODE %-12s | en_mask(N,S,W,E)=%0b%0b%0b%0b | N_out=0x%04x S_out=0x%04x W_out=0x%04x E_out=0x%04x",
                            mode_names[m], north_enable_o, south_enable_o, west_enable_o, east_enable_o,
                            north_enable_o ? north_data_o : 16'h0000,
                            south_enable_o ? south_data_o : 16'h0000,
                            west_enable_o  ? west_data_o  : 16'h0000,
                            east_enable_o  ? east_data_o  : 16'h0000);
                    end
                    // Assert every enabled destination receives src_words[s]
                    if (north_enable_o && north_data_o !== 16'(src_words[s])) $error("Mismatch N");
                    if (south_enable_o && south_data_o !== 16'(src_words[s])) $error("Mismatch S");
                    if (west_enable_o  && west_data_o  !== 16'(src_words[s])) $error("Mismatch W");
                    if (east_enable_o  && east_data_o  !== 16'(src_words[s])) $error("Mismatch E");
                end
            end
        end

        $display("\n=== 2. PRIORITY ARBITRATION WHEN MULTIPLE INPUT ENABLES ARE HIGH (MODE=0:ALL) ===");
        router_mode = 4'd0;
        // All 4 enabled: North (0x1111) should win
        {north_enable_i, south_enable_i, west_enable_i, east_enable_i} = 4'b1111; #1;
        $display("in_en=1111 -> data_out=0x%04x (expected 0x1111 NORTH)", north_data_o);
        // South + West + East: South (0x2222) should win
        {north_enable_i, south_enable_i, west_enable_i, east_enable_i} = 4'b0111; #1;
        $display("in_en=0111 -> data_out=0x%04x (expected 0x2222 SOUTH)", north_data_o);
        // West + East: West (0x3333) should win
        {north_enable_i, south_enable_i, west_enable_i, east_enable_i} = 4'b0011; #1;
        $display("in_en=0011 -> data_out=0x%04x (expected 0x3333 WEST)", north_data_o);
        // East only: East (0x4444) should win
        {north_enable_i, south_enable_i, west_enable_i, east_enable_i} = 4'b0001; #1;
        $display("in_en=0001 -> data_out=0x%04x (expected 0x4444 EAST)", north_data_o);

        $display("\n=== 3. IDLE INPUTS (ALL *_enable_i = 0) BEHAVIOR ===");
        {north_enable_i, south_enable_i, west_enable_i, east_enable_i} = 4'b0000; #1;
        $display("in_en=0000 (MODE=0:ALL) -> en_mask=%0b%0b%0b%0b data_out=%0d (0x%04x, sentinel 10101)",
            north_enable_o, south_enable_o, west_enable_o, east_enable_o, north_data_o, north_data_o);
        $finish;
    end
endmodule
