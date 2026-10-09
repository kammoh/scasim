// Two-stage pipe. Neither side has a ready signal: every cycle with valid is a transfer.
module stream_pipe (
    input  logic        clk,
    input  logic        rst,
    input  logic        in_valid,
    input  logic [15:0] in_data,
    output logic        out_valid,
    output logic [15:0] out_data
);
  logic        v1;
  logic [15:0] d1;
  always_ff @(posedge clk) begin
    if (rst) begin
      v1 <= 1'b0;
      out_valid <= 1'b0;
    end else begin
      v1 <= in_valid;
      out_valid <= v1;
    end
    d1 <= in_data;
    out_data <= d1;
  end
endmodule
