// Test top. Prefix c_: FIFO with combinational ready. r_: FIFO with registered ready.
// p_: pipe without ready. o_: pair reorder.
module stream_top (
    input logic clk,
    input logic rst,

    input  logic        c_in_valid,
    output logic        c_in_ready,
    input  logic [15:0] c_in_data,
    input  logic        c_in_last,
    input  logic [ 7:0] c_in_vec_0,
    input  logic [ 7:0] c_in_vec_1,
    output logic        c_out_valid,
    input  logic        c_out_ready,
    output logic [15:0] c_out_data,
    output logic        c_out_last,
    output logic [ 7:0] c_out_vec_0,
    output logic [ 7:0] c_out_vec_1,

    input  logic        r_in_valid,
    output logic        r_in_ready,
    input  logic [15:0] r_in_data,
    input  logic        r_in_last,
    input  logic [ 7:0] r_in_vec_0,
    input  logic [ 7:0] r_in_vec_1,
    output logic        r_out_valid,
    input  logic        r_out_ready,
    output logic [15:0] r_out_data,
    output logic        r_out_last,
    output logic [ 7:0] r_out_vec_0,
    output logic [ 7:0] r_out_vec_1,

    input  logic        p_in_valid,
    input  logic [15:0] p_in_data,
    output logic        p_out_valid,
    output logic [15:0] p_out_data,

    input  logic        o_in_valid,
    output logic        o_in_ready,
    input  logic [15:0] o_in_data,
    output logic        o_out_valid,
    input  logic        o_out_ready,
    output logic [15:0] o_out_data
);
  stream_fifo #(
      .DEPTH(4),
      .REG_READY(1'b0)
  ) fifo_c (
      .clk,
      .rst,
      .in_valid(c_in_valid),
      .in_ready(c_in_ready),
      .in_data(c_in_data),
      .in_last(c_in_last),
      .in_vec_0(c_in_vec_0),
      .in_vec_1(c_in_vec_1),
      .out_valid(c_out_valid),
      .out_ready(c_out_ready),
      .out_data(c_out_data),
      .out_last(c_out_last),
      .out_vec_0(c_out_vec_0),
      .out_vec_1(c_out_vec_1)
  );

  stream_fifo #(
      .DEPTH(4),
      .REG_READY(1'b1)
  ) fifo_r (
      .clk,
      .rst,
      .in_valid(r_in_valid),
      .in_ready(r_in_ready),
      .in_data(r_in_data),
      .in_last(r_in_last),
      .in_vec_0(r_in_vec_0),
      .in_vec_1(r_in_vec_1),
      .out_valid(r_out_valid),
      .out_ready(r_out_ready),
      .out_data(r_out_data),
      .out_last(r_out_last),
      .out_vec_0(r_out_vec_0),
      .out_vec_1(r_out_vec_1)
  );

  stream_pipe pipe_p (
      .clk,
      .rst,
      .in_valid(p_in_valid),
      .in_data(p_in_data),
      .out_valid(p_out_valid),
      .out_data(p_out_data)
  );

  stream_reorder reorder_o (
      .clk,
      .rst,
      .in_valid(o_in_valid),
      .in_ready(o_in_ready),
      .in_data(o_in_data),
      .out_valid(o_out_valid),
      .out_ready(o_out_ready),
      .out_data(o_out_data)
  );
endmodule
