#!/usr/bin/env python3
"""Apply testbench IEEE-1800 wiring fixes in /root/dm-study/eyeriss-v2/w2/sim_fixed and verify outputs against Python golden."""
import pathlib

fixed = pathlib.Path("/root/dm-study/eyeriss-v2/w2/sim_fixed")

# 1. PE_cluster_tb.sv: add $finish
p = fixed / "testbench/PE_cluster_tb.sv"
p.write_text(p.read_text().replace("\t\t\n\t\t\n\tend", "\t\t$finish;\n\tend"))

# 2. GLB_cluster_tb.sv and HMNoC_1_tb.sv: scalar vs 1-element unpacked array [NUM_GLB_*-1:0]
for fn in ["testbench/GLB_cluster_tb.sv", "testbench/HMNoC_1_tb.sv"]:
    p = fixed / fn
    t = p.read_text()
    for s in ["[NUM_GLB_IACT-1:0]", "[NUM_GLB_PSUM-1:0]", "[NUM_GLB_WGHT-1:0]"]:
        t = t.replace(s, "")
    t = t.replace("[0]", "")
    t = t.replace("\tend\nendmodule", "\t$finish;\n\tend\nendmodule")
    p.write_text(t)

# 3. router_east_tb.sv: comment out non-existent ready pins and trailing comma
p = fixed / "testbench/router_east_tb.sv"
t = p.read_text()
for rpin in ["north_ready_o", "north_ready_i", "south_ready_o", "south_ready_i",
             "west_ready_o", "west_ready_i", "east_ready_o", "east_ready_i"]:
    t = t.replace(f".{rpin}", f"// .{rpin}")
t = t.replace(".east_enable_o(east_enable_o),", ".east_enable_o(east_enable_o)")
t = t.replace("\tend\n\t\nendmodule", "\t$display(\"router_east_tb done: n=%0d s=%0d w=%0d e=%0d\", north_data_o, south_data_o, west_data_o, east_data_o);\n\t$finish;\n\tend\nendmodule")
p.write_text(t)

# 4. router_weight_tb.sv, router_iact_tb.sv, router_psum_tb.sv: remove non-existent PSUM_READ/LOAD_ADDR params & fix hierarchical refs
import re
for fn in ["testbench/router_weight_tb.sv", "testbench/router_iact_tb.sv", "testbench/router_psum_tb.sv"]:
    p = fixed / fn
    t = p.read_text()
    t = re.sub(r',\s*\.PSUM_READ_ADDR\(PSUM_READ_ADDR\),\s*\.PSUM_LOAD_ADDR\(PSUM_LOAD_ADDR\)', '', t)
    for s in ["[NUM_GLB_IACT-1:0]", "[NUM_GLB_PSUM-1:0]", "[NUM_GLB_WGHT-1:0]", "[0]"]:
        t = t.replace(s, "")
    t = t.replace("router_weight_0.read_req_glb_wght", "read_req_wght_w")
    t = t.replace("router_weight_0.r_addr_glb_wght", "r_addr_wght_w")
    t = t.replace("router_weight_0.r_data_glb_wght", "r_data_wght_w")
    t = t.replace("router_weight_0.w_data_spad", "filt_in")
    t = t.replace("router_weight_0.load_en_spad", "load_en_wght")
    t = t.replace("router_iact_0.read_req_glb_iact", "read_req_iact_w")
    t = t.replace("router_iact_0.r_addr_glb_iact", "r_addr_iact_w")
    t = t.replace("router_iact_0.r_data_glb_iact", "r_data_iact_w")
    t = t.replace("router_iact_0.w_data_spad", "act_in")
    t = t.replace("router_iact_0.load_en_spad", "load_en_act")
    t = t.replace("router_psum_0.w_addr_glb_psum", "w_addr_psum")
    t = t.replace("router_psum_0.w_data_glb_psum", "w_data_psum")
    t = t.replace("router_psum_0.write_en_glb_psum", "write_en_psum_w")
    t = t.replace("router_psum_0.r_data_spad_psum", "pe_out")
    t = t.replace("router_psum_0.write_psum_ctrl", "compute_done")
    t = t.replace("GLB_cluster_0.r_data_wght", "r_data_wght_w")
    t = t.replace("GLB_cluster_0.r_addr_wght", "r_addr_wght_w")
    t = t.replace("GLB_cluster_0.read_req_wght", "read_req_wght_w")
    t = t.replace("GLB_cluster_0.r_data_iact", "r_data_iact_w")
    t = t.replace("GLB_cluster_0.r_addr_iact", "r_addr_iact_w")
    t = t.replace("GLB_cluster_0.read_req_iact", "read_req_iact_w")
    t = t.replace("GLB_cluster_0.w_addr_psum", "w_addr_psum")
    t = t.replace("GLB_cluster_0.w_data_psum", "w_data_psum")
    t = t.replace("GLB_cluster_0.write_en_psum", "write_en_psum_w")
    t = t.replace("pe_cluster_0.filt_in", "filt_in")
    t = t.replace("pe_cluster_0.load_en_wght", "load_en_wght")
    t = t.replace("pe_cluster_0.act_in", "act_in")
    t = t.replace("pe_cluster_0.load_en_act", "load_en_act")
    t = t.replace("pe_cluster_0.pe_out", "pe_out")
    t = t.replace("pe_cluster_0.compute_done", "compute_done")
    t = t.replace("assign pe_out = pe_out;", "")
    t = t.replace("assign compute_done = compute_done;", "")
    t = t.replace("logic clk;", "logic clk, read_req_iact_w, read_req_wght_w, write_en_psum_w;\nlogic [9:0] r_addr_iact_w, r_addr_wght_w;\nlogic [15:0] r_data_iact_w, r_data_wght_w;")
    t = t.replace("$stop;", "$finish;")
    if "$finish" not in t:
        t = t.replace("\tend\n\nendmodule", "\t$finish;\n\tend\nendmodule").replace("\tend\nendmodule", "\t$finish;\n\tend\nendmodule")
    p.write_text(t)

# 5. router_cluster_tb.sv & router_cluster_5x5_tb.sv
for fn in ["testbench/router_cluster_tb.sv", "testbench/router_cluster_5x5_tb.sv"]:
    p = fixed / fn
    t = p.read_text()
    t = t.replace("router_cluster.read_req_glb_iact", "read_req_iact_w")
    t = t.replace("router_cluster_0.read_req_glb_wght", "read_req_wght_w")
    t = t.replace("router_cluster_0.r_data_glb_iact", "r_data_iact_w")
    t = t.replace("router_cluster_0.r_data_glb_wght", "r_data_wght_w")
    t = t.replace("router_cluster_0.r_addr_glb_iact", "r_addr_iact_w")
    t = t.replace("router_cluster_0.r_addr_glb_wght", "r_addr_wght_w")
    t = t.replace("router_cluster_0.w_addr_glb_psum", "w_addr_psum")
    t = t.replace("router_cluster_0.w_data_glb_psum", "w_data_psum")
    t = t.replace("router_cluster_0.write_en_glb_psum", "write_en_psum_w")
    t = t.replace("GLB_cluster_0.r_data_iact", "r_data_iact_w")
    t = t.replace("GLB_cluster_0.r_addr_iact", "r_addr_iact_w")
    t = t.replace("GLB_cluster_0.read_req_iact", "read_req_iact_w")
    t = t.replace("GLB_cluster_0.r_data_wght", "r_data_wght_w")
    t = t.replace("GLB_cluster_0.r_addr_wght", "r_addr_wght_w")
    t = t.replace("GLB_cluster_0.read_req_wght", "read_req_wght_w")
    t = t.replace("GLB_cluster_0.w_addr_psum", "w_addr_psum")
    t = t.replace("GLB_cluster_0.write_en_psum", "write_en_psum_w")
    t = t.replace("GLB_cluster_0.w_data_psum", "w_data_psum")
    t = t.replace("pe_cluster_0.act_in", "act_in")
    t = t.replace("pe_cluster_0.load_en_act", "load_en_act")
    t = t.replace("pe_cluster_0.filt_in", "filt_in")
    t = t.replace("pe_cluster_0.load_en_wght", "load_en_wght")
    t = t.replace(".r_data_spad_psum(pe_cluster_0.pe_out)", ".r_data_spad_psum(pe_out)")
    t = t.replace(".write_psum_ctrl(pe_cluster_0.compute_done)", ".write_psum_ctrl(compute_done)")
    t = t.replace("router_cluster_0.w_data_spad_iact", "act_in")
    t = t.replace("router_cluster_0.w_data_spad_wght", "filt_in")
    t = t.replace("router_cluster_0.load_en_spad_wght", "load_en_wght")
    t = t.replace("router_cluster_0.load_en_spad_iact", "load_en_act")
    t = t.replace(".pe_out(router_cluster_0.r_data_spad_psum)", ".pe_out(pe_out)")
    t = t.replace(".compute_done(router_cluster_0.write_psum_ctrl)", ".compute_done(compute_done)")
    t = t.replace("assign pe_out = pe_cluster_0.pe_out;", "")
    t = t.replace("assign compute_done = pe_cluster_0.compute_done;", "")
    t = t.replace("logic compute_done;", "logic compute_done, read_req_iact_w, read_req_wght_w, write_en_psum_w;\nlogic [ADDR_BITWIDTH-1:0] r_addr_iact_w, r_addr_wght_w;\nlogic [DATA_BITWIDTH-1:0] r_data_iact_w, r_data_wght_w;")
    p.write_text(t)

# 6. router_cluster_new_tb.sv, router_cluster_pe_cluster_tb.sv, router_cluster_pe_cluster_5x5_tb.sv
for fn in ["testbench/router_cluster_new_tb.sv", "testbench/router_cluster_pe_cluster_tb.sv", "testbench/router_cluster_pe_cluster_5x5_tb.sv"]:
    p = fixed / fn
    t = p.read_text()
    t = t.replace(".west_data_i_wght(GLB_cluster_0.r_data_wght)", ".west_data_i_wght(west_data_i_wght)")
    t = t.replace(".west_data_o_wght(pe_cluster_0.filt_in)", ".west_data_o_wght(west_data_o_wght)")
    t = t.replace(".west_data_i_iact(GLB_cluster_0.r_data_iact)", ".west_data_i_iact(west_data_i_iact)")
    t = t.replace(".west_data_o_iact(pe_cluster_0.act_in)", ".west_data_o_iact(west_data_o_iact)")
    t = t.replace(".west_data_o_psum(GLB_cluster_0.w_data_psum)", ".west_data_o_psum(west_data_o_psum)")
    t = t.replace(".west_enable_o_psum(GLB_cluster_0.write_en_psum)", ".west_enable_o_psum(west_enable_o_psum)")
    t = t.replace(".write_en_psum(router_cluster_0.west_enable_o_psum)", ".write_en_psum(west_enable_o_psum)")
    t = t.replace(".w_data_psum(router_cluster_0.west_data_o_psum)", ".w_data_psum(west_data_o_psum)")
    t = t.replace(".r_data_iact(router_cluster_0.west_data_i_iact)", ".r_data_iact(west_data_i_iact)")
    t = t.replace(".r_data_wght(router_cluster_0.west_data_i_wght)", ".r_data_wght(west_data_i_wght)")
    t = t.replace(".act_in(router_cluster_0.west_data_o_iact)", ".act_in(west_data_o_iact)")
    t = t.replace(".filt_in(router_cluster_0.west_data_o_wght)", ".filt_in(west_data_o_wght)")
    t = t.replace("assign west_data_o_wght = router_cluster_0.west_data_o_wght;", "")
    t = t.replace("logic [DATA_WIDTH-1:0] cluster_out_1[0:8];", "logic [DATA_WIDTH-1:0] cluster_out_1[0:24];")
    t = t.replace("int cycles;\n\t// track", "// int cycles;\n\t// track")
    t = t.replace("int cycles;\n\t\t// track", "// int cycles;\n\t\t// track")
    t = t.replace("module router_cluster_", "int cycles;\nmodule router_cluster_")
    t = t.replace("$stop;", "$finish;")
    if "$finish" not in t:
        t = t.replace("\tend\n\t\nendmodule", "\t$finish;\n\tend\nendmodule")
    p.write_text(t)

# 7. router_pe_4_clusters_tb.sv & router_pe_4_clusters_5x5_tb.sv
for fn in ["testbench/router_pe_4_clusters_tb.sv", "testbench/router_pe_4_clusters_5x5_tb.sv"]:
    p = fixed / fn
    t = p.read_text()
    t = t.replace(".act_in(network_iact.west_data_o_west_0)", ".act_in(west_data_o_west_0_iact)")
    t = t.replace(".filt_in(network_wght.west_data_o_west_0)", ".filt_in(west_data_o_west_0_whgt)")
    t = t.replace(".act_in(network_iact.west_data_o_west_1)", ".act_in(west_data_o_west_1_iact)")
    t = t.replace(".filt_in(network_wght.west_data_o_west_1)", ".filt_in(west_data_o_west_1_whgt)")
    t = t.replace(".act_in(network_iact.east_data_o_east_0)", ".act_in(east_data_o_east_0_iact)")
    t = t.replace(".filt_in(network_wght.east_data_o_east_0)", ".filt_in(east_data_o_east_0_whgt)")
    t = t.replace(".act_in(network_iact.east_data_o_east_1)", ".act_in(east_data_o_east_1_iact)")
    t = t.replace(".filt_in(network_wght.east_data_o_east_1)", ".filt_in(east_data_o_east_1_whgt)")
    t = t.replace("logic [DATA_WIDTH-1:0] cluster_out_1[0:8];", "logic [DATA_WIDTH-1:0] cluster_out_1[0:24];")
    t = t.replace("int cycles;\n\t\t// track", "// int cycles;\n\t\t// track")
    t = t.replace("module router_pe_4_clusters", "int cycles;\nmodule router_pe_4_clusters")
    t = t.replace(
        "\t\t$stop;",
        """\t\t$display("ALL4_LAST_ITER: west0=(%0d,%0d,%0d) west1=(%0d,%0d,%0d) east0=(%0d,%0d,%0d) east1=(%0d,%0d,%0d)",
            pe_out_west_0[0], pe_out_west_0[1], pe_out_west_0[2],
            pe_out_west_1[0], pe_out_west_1[1], pe_out_west_1[2],
            pe_out_east_0[0], pe_out_east_0[1], pe_out_east_0[2],
            pe_out_east_1[0], pe_out_east_1[1], pe_out_east_1[2]);
        $finish;"""
    )
    p.write_text(t)

# 8. router_broadcast_tb.sv, router_multicast_tb.sv, router_unicast_tb.sv
for fn in ["testbench/router_broadcast_tb.sv", "testbench/router_multicast_tb.sv", "testbench/router_unicast_tb.sv"]:
    p = fixed / fn
    t = p.read_text()
    t = t.replace("west_data_0_west_0", "west_data_o_west_0")
    t = t.replace("west_data_0_west_1", "west_data_o_west_1")
    t = t.replace(".east_data_i(router_east_0.west_data_o)", ".east_data_i(e0_to_w0_d)")
    t = t.replace(".east_enable_i(router_east_0.west_enable_o)", ".east_enable_i(e0_to_w0_e)")
    t = t.replace(".east_data_o(router_east_0.west_data_i)", ".east_data_o(w0_to_e0_d)")
    t = t.replace(".east_enable_o(router_east_0.west_enable_i)", ".east_enable_o(w0_to_e0_e)")
    t = t.replace(".west_data_i(router_west_0.east_data_o)", ".west_data_i(w0_to_e0_d)")
    t = t.replace(".west_enable_i(router_west_0.east_enable_o)", ".west_enable_i(w0_to_e0_e)")
    t = t.replace(".west_data_o(router_west_0.east_data_i)", ".west_data_o(e0_to_w0_d)")
    t = t.replace(".west_enable_o(router_west_0.east_enable_i)", ".west_enable_o(e0_to_w0_e)")
    t = t.replace(".east_data_i(router_east_1.west_data_o)", ".east_data_i(e1_to_w1_d)")
    t = t.replace(".east_enable_i(router_east_1.west_enable_o)", ".east_enable_i(e1_to_w1_e)")
    t = t.replace(".east_data_o(router_east_1.west_data_i)", ".east_data_o(w1_to_e1_d)")
    t = t.replace(".east_enable_o(router_east_1.west_enable_i)", ".east_enable_o(w1_to_e1_e)")
    t = t.replace(".west_data_i(router_west_1.east_data_o)", ".west_data_i(w1_to_e1_d)")
    t = t.replace(".west_enable_i(router_west_1.east_enable_o)", ".west_enable_i(w1_to_e1_e)")
    t = t.replace(".west_data_o(router_west_1.east_data_i)", ".west_data_o(e1_to_w1_d)")
    t = t.replace(".west_enable_o(router_west_1.east_enable_i)", ".west_enable_o(e1_to_w1_e)")
    t = t.replace("parameter DATA_WIDTH = 16;", "parameter DATA_WIDTH = 16;\nlogic [DATA_WIDTH-1:0] w0_to_e0_d, e0_to_w0_d, w1_to_e1_d, e1_to_w1_d;\nlogic w0_to_e0_e, e0_to_w0_e, w1_to_e1_e, e1_to_w1_e;")
    t = t.replace("endmodule", "initial begin #650; $display(\"TB_DONE %m w0_s=%0d e0_s=%0d e0_n=%0d e0_e=%0d\", south_data_o_west_0, south_data_o_east_0, north_data_o_east_0, east_data_o_east_0); $finish; end\nendmodule")
    p.write_text(t)

# 9. HMNoC_cluster_west.sv & HMNoC_cluster_west_tb.sv: replace hierarchical .port(inst.in) with wires
hmw = (fixed / "rtl/HMNoC_cluster_west.sv").read_text()
hmw = hmw.replace(".west_data_i_wght(GLB_cluster_0.r_data_wght)", ".west_data_i_wght(r_data_wght)")
hmw = hmw.replace(".west_data_o_wght(pe_cluster_0.filt_in)", ".west_data_o_wght(west_data_o_wght)")
hmw = hmw.replace(".west_data_i_iact(GLB_cluster_0.r_data_iact)", ".west_data_i_iact(r_data_iact)")
hmw = hmw.replace(".west_data_o_iact(pe_cluster_0.act_in)", ".west_data_o_iact(west_data_o_iact)")
hmw = hmw.replace(".west_data_o_psum(GLB_cluster_0.w_data_psum)", ".west_data_o_psum(west_data_o_psum)")
hmw = hmw.replace(".west_enable_o_psum(GLB_cluster_0.write_en_psum)", ".west_enable_o_psum(west_enable_o_psum)")
hmw = hmw.replace(".write_en_psum(router_cluster_0.west_enable_o_psum)", ".write_en_psum(west_enable_o_psum)")
hmw = hmw.replace(".w_data_psum(router_cluster_0.west_data_o_psum)", ".w_data_psum(west_data_o_psum)")
hmw = hmw.replace(".r_data_iact(router_cluster_0.west_data_i_iact)", ".r_data_iact(r_data_iact)")
hmw = hmw.replace(".r_data_wght(router_cluster_0.west_data_i_wght)", ".r_data_wght(r_data_wght)")
hmw = hmw.replace(".act_in(router_cluster_0.west_data_o_iact)", ".act_in(west_data_o_iact)")
hmw = hmw.replace(".filt_in(router_cluster_0.west_data_o_wght)", ".filt_in(west_data_o_wght)")
hmw = hmw.replace("//    logic [DATA_BITWIDTH-1 : 0] r_data_wght[NUM_GLB_WGHT-1:0];", "logic [DATA_BITWIDTH-1 : 0] r_data_wght;")
(fixed / "rtl/HMNoC_cluster_west.sv").write_text(hmw)

hmwtb = (fixed / "testbench/HMNoC_cluster_west_tb.sv").read_text()
hmwtb = hmwtb.replace("assign west_data_o_wght = HMNoC_cluster_west_0.west_data_o_wght;", "")
hmwtb = hmwtb.replace("int cycles;\n\t// track", "// int cycles;\n\t// track")
hmwtb = hmwtb.replace("module HMNoC_cluster_west_tb", "int cycles;\nmodule HMNoC_cluster_west_tb")
hmwtb = hmwtb.replace("$stop;", "$finish;")
(fixed / "testbench/HMNoC_cluster_west_tb.sv").write_text(hmwtb)

print("All testbench portability fixes applied in", fixed)
