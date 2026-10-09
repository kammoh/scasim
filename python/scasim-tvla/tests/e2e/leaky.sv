// A masked pipeline with an optional planted leak, for the end-to-end tests.
//
// The input goes through four stages as a share and a mask (share = data ^ mask), so no stage
// carries the data in the clear. A xorshift generator makes the masks and keeps the toggle
// counts noisy. With LEAK = 1, the register `leak` copies the unmasked value of stage
// STAGE-1 at edge E_STAGE and clears one edge later. That is the planted leak: the number of
// toggles at those two edges follows the Hamming weight of the data.
//
// Edge E_0 is the edge that accepts the input. Stage k loads at E_k.
module leaky_noise (
    input  logic        clk,
    output logic [15:0] mask
);
  logic [31:0] rng = 32'h1234_5678;
  logic [31:0] r1, r2, r3;
  assign r1 = rng ^ (rng << 13);
  assign r2 = r1 ^ (r1 >> 17);
  assign r3 = r2 ^ (r2 << 5);
  always_ff @(posedge clk) rng <= r3;
  assign mask = rng[15:0];
endmodule

module leaky #(
    parameter int LEAK  = 1,
    parameter int STAGE = 2
) (
    input  logic        clk,
    input  logic        in_valid,
    input  logic [15:0] in_data,
    output logic        out_valid,
    output logic [15:0] out_share,
    output logic [15:0] out_mask
);
  logic [15:0] rmask;
  leaky_noise u_noise (.clk(clk), .mask(rmask));

  logic [15:0] share[4];
  logic [15:0] mask [4];
  logic        valid[4];
  logic [15:0] leak = 16'h0000;

  initial begin
    for (int i = 0; i < 4; i++) begin
      share[i] = 16'h0;
      mask[i]  = 16'h0;
      valid[i] = 1'b0;
    end
  end

  always_ff @(posedge clk) begin
    share[0] <= in_data ^ rmask;
    mask[0]  <= rmask;
    valid[0] <= in_valid;
    for (int i = 1; i < 4; i++) begin
      share[i] <= share[i-1];
      mask[i]  <= mask[i-1];
      valid[i] <= valid[i-1];
    end
    leak <= (LEAK != 0 && valid[STAGE-1]) ? (share[STAGE-1] ^ mask[STAGE-1]) : 16'h0000;
  end

  assign out_valid = valid[3];
  assign out_share = share[3];
  assign out_mask  = mask[3];
endmodule
