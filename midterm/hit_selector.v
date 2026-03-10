`timescale 1ns / 1ps

module hit_selector #(
    parameter integer X_W   = 5,
    parameter integer Y_W   = 8,
    parameter integer X_MIN = 0,
    parameter integer X_MAX = 19,
    parameter integer Y_MIN = 120,
    parameter integer Y_MAX = 150
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 valid_in,
    input  wire [X_W-1:0]       x_in,
    input  wire [Y_W-1:0]       y_in,
    output reg                  valid_out,
    output reg                  label_out
);

    wire in_roi;

    assign in_roi = (x_in >= X_MIN) && (x_in <= X_MAX) &&
                    (y_in >= Y_MIN) && (y_in <= Y_MAX);

    always @(posedge clk) begin
        if (rst) begin
            valid_out <= 1'b0;
            label_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
            if (valid_in)
                label_out <= in_roi;
            else
                label_out <= 1'b0;
        end
    end

endmodule
