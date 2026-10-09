// FIFO with a valid/ready input and output. Fields: data (packed), last (1 bit),
// vec_0 and vec_1 (a flattened array). With REG_READY the input ready is a register.
module stream_fifo #(
    parameter int DEPTH = 4,
    parameter bit REG_READY = 1'b0
) (
    input  logic        clk,
    input  logic        rst,
    input  logic        in_valid,
    output logic        in_ready,
    input  logic [15:0] in_data,
    input  logic        in_last,
    input  logic [ 7:0] in_vec_0,
    input  logic [ 7:0] in_vec_1,
    output logic        out_valid,
    input  logic        out_ready,
    output logic [15:0] out_data,
    output logic        out_last,
    output logic [ 7:0] out_vec_0,
    output logic [ 7:0] out_vec_1
);
  localparam int AW = $clog2(DEPTH);
  logic [32:0] mem[DEPTH];
  logic [AW-1:0] rd, wr;
  logic [AW:0] cnt;
  logic ready_q;

  assign in_ready  = REG_READY ? ready_q : (cnt != (AW + 1)'(DEPTH));
  assign out_valid = cnt != 0;
  wire push = in_valid && in_ready;
  wire pop = out_valid && out_ready;
  wire [AW:0] cnt_next = cnt + (AW + 1)'(push) - (AW + 1)'(pop);
  assign {out_last, out_vec_1, out_vec_0, out_data} = mem[rd];

  always_ff @(posedge clk) begin
    if (rst) begin
      rd <= '0;
      wr <= '0;
      cnt <= '0;
      ready_q <= 1'b0;
    end else begin
      if (push) begin
        mem[wr] <= {in_last, in_vec_1, in_vec_0, in_data};
        wr <= wr + 1'b1;
      end
      if (pop) rd <= rd + 1'b1;
      cnt <= cnt_next;
      ready_q <= cnt_next != (AW + 1)'(DEPTH);
    end
  end
endmodule
