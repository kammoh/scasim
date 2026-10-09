// Tiny design for the session tests: q loads d when load is set, and cnt counts loads.
module tiny (
    input  logic       clk,
    input  logic       load,
    input  logic [7:0] d,
    output logic [7:0] q = 8'h00,
    output logic [7:0] cnt = 8'h00
);
  always_ff @(posedge clk) begin
    if (load) begin
      q   <= d;
      cnt <= cnt + 8'd1;
    end
  end
endmodule
