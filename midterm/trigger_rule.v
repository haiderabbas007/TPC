`timescale 1ns / 1ps

module trigger_rule #(
    parameter integer ROI_ACTIVE_ROWS_W   = 4,
    parameter integer LONGEST_ROW_RUN_W   = 8,
    parameter integer ROI_LONGEST_Y_RUN_W = 8,
    parameter integer LONGEST_Y_RUN_W     = 8
)(
    input  wire [ROI_ACTIVE_ROWS_W-1:0]   roi_active_rows,
    input  wire [LONGEST_ROW_RUN_W-1:0]   longest_row_run,
    input  wire [ROI_LONGEST_Y_RUN_W-1:0] roi_longest_y_run,
    input  wire [LONGEST_Y_RUN_W-1:0]     longest_y_run,
    output wire                           trigger_out,
    output wire                           branch1_out,
    output wire                           branch2_out
);

    assign branch1_out =
        (roi_active_rows   <= 4'd2) &&
        (longest_row_run   >= 8'd2) &&
        (roi_longest_y_run >= 8'd2);

    assign branch2_out =
        (roi_active_rows <= 4'd4) &&
        (longest_y_run   >= 8'd10);

    assign trigger_out = branch1_out || branch2_out;

endmodule
