// STAND-IN written by me for study E6 (the repository has no `Mux` module although PE_Cluster.v instantiates one).
// Behaviour guessed from the instantiation: DATA_NUM packed inputs, `sel` picks one.
module Mux #(parameter DATA_WIDTH = 16, parameter DATA_NUM = 2)(
    input  [DATA_WIDTH*DATA_NUM-1:0] data_in_packed,
    input  [$clog2(DATA_NUM)-1:0]    sel,
    output [DATA_WIDTH-1:0]          data_out
);
    assign data_out = data_in_packed[sel*DATA_WIDTH +: DATA_WIDTH];
endmodule
