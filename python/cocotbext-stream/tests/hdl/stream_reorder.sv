// Takes two items (a, b) and emits them in the order b, a.
module stream_reorder (
    input  logic        clk,
    input  logic        rst,
    input  logic        in_valid,
    output logic        in_ready,
    input  logic [15:0] in_data,
    output logic        out_valid,
    input  logic        out_ready,
    output logic [15:0] out_data
);
  typedef enum logic [1:0] {
    TAKE_A,
    TAKE_B,
    EMIT_B,
    EMIT_A
  } state_t;
  state_t state;
  logic [15:0] a, b;

  assign in_ready  = state == TAKE_A || state == TAKE_B;
  assign out_valid = state == EMIT_B || state == EMIT_A;
  assign out_data  = state == EMIT_B ? b : a;

  always_ff @(posedge clk) begin
    if (rst) state <= TAKE_A;
    else
      case (state)
        TAKE_A:
        if (in_valid) begin
          a <= in_data;
          state <= TAKE_B;
        end
        TAKE_B:
        if (in_valid) begin
          b <= in_data;
          state <= EMIT_B;
        end
        EMIT_B: if (out_ready) state <= EMIT_A;
        EMIT_A: if (out_ready) state <= TAKE_A;
        default: state <= TAKE_A;
      endcase
  end
endmodule
