`timescale 1ns/1ns
// Non-intrusive probe compiled alongside unmodified sim/PE_tb.sv in work/
module pe_probe;
    integer idx;
    always @(posedge PE_tb.clk) begin
        if (PE_tb.dut.iact_counter_en)
            $display("[T=%0t] IACT_SPAD_WRITE: addr=%0d data=%0d", $time, PE_tb.dut.ifmap_write_cout, PE_tb.dut.iact_buffer_read_data);
        if (PE_tb.dut.weight_counter_en)
            $display("[T=%0t] WEIGHT_SPAD_WRITE: addr=%0d data=%0d", $time, PE_tb.dut.filter_write_cout, PE_tb.dut.weight_buffer_read_data);
        if (PE_tb.dut.pipe_en) begin
            $display("[T=%0t] PIPE: iread=%0d(%0d) wread=%0d(%0d) zf=%b | ri=%0d rw=%0d | mul=%0d mv=%b | paddr=%0d pdis=%b ptomux=%0d sum=%0d | pmux=%h reg_pmux=%h pwen=%b out=%h",
                $time,
                PE_tb.dut.ifmap_read_cout, PE_tb.dut.ifmap_spad_read_data,
                PE_tb.dut.filter_read_cout, PE_tb.dut.filter_spad_read_data,
                PE_tb.dut.zero_flag,
                PE_tb.dut.reg_iact_data, PE_tb.dut.reg_filter_data,
                PE_tb.dut.mult_data_out, PE_tb.dut.mul_out_valid,
                PE_tb.dut.psum_write_addr_fr, PE_tb.dut.psum_zero_flag,
                PE_tb.dut.psum_tomux, PE_tb.dut.sum_adder_out,
                PE_tb.dut.psum_mux_out, PE_tb.dut.reg_out_psum_mux,
                PE_tb.dut.psum_write_cnt_en_reg, PE_tb.data_psum_out);
        end
    end
    initial begin
        #378;
        $display("=== FINAL INTERNAL STATE OF PE_tb.dut AT T=378 ===");
        for (idx = 0; idx < 12; idx = idx + 1)
            $display("  ifmap_spad.regs[%0d] = %0d", idx, PE_tb.dut.ifmap_spad.regs[idx]);
        for (idx = 0; idx < 8; idx = idx + 1)
            $display("  filter_spad.mem[%0d] = %0d", idx, PE_tb.dut.filter_spad.mem[idx]);
        for (idx = 0; idx < 6; idx = idx + 1)
            $display("  psum_spad.regs[%0d]  = %0d (hex %h)", idx, PE_tb.dut.psum_spad.regs[idx], PE_tb.dut.psum_spad.regs[idx]);
        for (idx = 0; idx < 6; idx = idx + 1)
            $display("  output_psum_fifo.mem[%0d] = %h", idx, PE_tb.dut.output_psum_fifo.buffer.memory[idx]);
        $display("  psum_mux_out = %h, reg_out_psum_mux = %h",
            PE_tb.dut.psum_mux_out, PE_tb.dut.reg_out_psum_mux);
    end
endmodule
