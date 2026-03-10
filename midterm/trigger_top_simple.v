`timescale 1ns / 1ps

module trigger_top_simple #(
    parameter integer X_W   = 5,
    parameter integer Y_W   = 8,
    parameter integer X_MIN = 0,
    parameter integer X_MAX = 19,
    parameter integer Y_MIN = 120,
    parameter integer Y_MAX = 150
)(
    input  wire               clk,
    input  wire               rst,
    input  wire               valid_in,
    input  wire [X_W-1:0]     x_in,
    input  wire [Y_W-1:0]     y_in,
    input  wire               event_done,

    output wire               valid_out,
    output wire               label_out,

    output wire [3:0]         roi_active_rows_dbg,
    output wire [7:0]         longest_row_run_dbg,
    output wire [7:0]         roi_longest_y_run_dbg,
    output wire [7:0]         longest_y_run_dbg,
    output wire               branch1_dbg,
    output wire               branch2_dbg
);

    wire        feat_valid;
    wire [3:0]  roi_active_rows;
    wire [7:0]  longest_row_run;
    wire [7:0]  roi_longest_y_run;
    wire [7:0]  longest_y_run;

    feature_extractor_simple #(
        .X_W(X_W),
        .Y_W(Y_W),
        .X_MIN(X_MIN),
        .X_MAX(X_MAX),
        .Y_MIN(Y_MIN),
        .Y_MAX(Y_MAX)
    ) feat_u (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in),
        .x_in(x_in),
        .y_in(y_in),
        .event_done(event_done),
        .valid_out(feat_valid),
        .roi_active_rows(roi_active_rows),
        .longest_row_run(longest_row_run),
        .roi_longest_y_run(roi_longest_y_run),
        .longest_y_run(longest_y_run)
    );

    trigger_rule rule_u (
        .roi_active_rows(roi_active_rows),
        .longest_row_run(longest_row_run),
        .roi_longest_y_run(roi_longest_y_run),
        .longest_y_run(longest_y_run),
        .trigger_out(label_out),
        .branch1_out(branch1_dbg),
        .branch2_out(branch2_dbg)
    );

    assign valid_out              = feat_valid;
    assign roi_active_rows_dbg    = roi_active_rows;
    assign longest_row_run_dbg    = longest_row_run;
    assign roi_longest_y_run_dbg  = roi_longest_y_run;
    assign longest_y_run_dbg      = longest_y_run;

endmodule
