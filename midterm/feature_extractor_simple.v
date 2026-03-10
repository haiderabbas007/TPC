`timescale 1ns / 1ps

module feature_extractor_simple #(
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

    output reg                valid_out,
    output reg  [3:0]         roi_active_rows,
    output reg  [7:0]         longest_row_run,
    output reg  [7:0]         roi_longest_y_run,
    output reg  [7:0]         longest_y_run
);

    reg [19:0] roi_rows_seen;

    reg [Y_W-1:0] prev_y_all;
    reg [Y_W-1:0] prev_y_roi;

    reg           have_prev_all;
    reg           have_prev_roi;

    reg [7:0]     cur_run_all;
    reg [7:0]     best_run_all;

    reg [7:0]     cur_run_roi;
    reg [7:0]     best_run_roi;

    reg           in_roi;

    integer i;
    integer row_count;

    always @(posedge clk) begin
        if (rst) begin
            roi_rows_seen      <= 20'd0;

            prev_y_all         <= {Y_W{1'b0}};
            prev_y_roi         <= {Y_W{1'b0}};
            have_prev_all      <= 1'b0;
            have_prev_roi      <= 1'b0;

            cur_run_all        <= 8'd0;
            best_run_all       <= 8'd0;

            cur_run_roi        <= 8'd0;
            best_run_roi       <= 8'd0;

            valid_out          <= 1'b0;
            roi_active_rows    <= 4'd0;
            longest_row_run    <= 8'd0;
            roi_longest_y_run  <= 8'd0;
            longest_y_run      <= 8'd0;
        end else begin
            valid_out <= 1'b0;

            if (valid_in) begin
                in_roi = (x_in >= X_MIN) && (x_in <= X_MAX) &&
                         (y_in >= Y_MIN) && (y_in <= Y_MAX);

                // Track ROI-active rows
                if (in_roi)
                    roi_rows_seen[x_in] <= 1'b1;

                // Global longest y-run
                if (!have_prev_all) begin
                    have_prev_all <= 1'b1;
                    prev_y_all    <= y_in;
                    cur_run_all   <= 8'd1;
                    best_run_all  <= 8'd1;
                end else begin
                    if (y_in == prev_y_all + 1'b1)
                        cur_run_all <= cur_run_all + 1'b1;
                    else
                        cur_run_all <= 8'd1;

                    prev_y_all <= y_in;

                    if (y_in == prev_y_all + 1'b1) begin
                        if ((cur_run_all + 1'b1) > best_run_all)
                            best_run_all <= cur_run_all + 1'b1;
                    end else begin
                        if (best_run_all < 8'd1)
                            best_run_all <= 8'd1;
                    end
                end

                // ROI longest y-run
                if (in_roi) begin
                    if (!have_prev_roi) begin
                        have_prev_roi <= 1'b1;
                        prev_y_roi    <= y_in;
                        cur_run_roi   <= 8'd1;
                        best_run_roi  <= 8'd1;
                    end else begin
                        if (y_in == prev_y_roi + 1'b1)
                            cur_run_roi <= cur_run_roi + 1'b1;
                        else
                            cur_run_roi <= 8'd1;

                        prev_y_roi <= y_in;

                        if (y_in == prev_y_roi + 1'b1) begin
                            if ((cur_run_roi + 1'b1) > best_run_roi)
                                best_run_roi <= cur_run_roi + 1'b1;
                        end else begin
                            if (best_run_roi < 8'd1)
                                best_run_roi <= 8'd1;
                        end
                    end
                end
            end

            if (event_done) begin
                row_count = 0;
                for (i = 0; i < 20; i = i + 1) begin
                    if (roi_rows_seen[i])
                        row_count = row_count + 1;
                end

                roi_active_rows   <= row_count[3:0];
                roi_longest_y_run <= best_run_roi;
                longest_y_run     <= best_run_all;

                // Temporary approximation for simple pipeline
                longest_row_run   <= best_run_roi;

                valid_out         <= 1'b1;

                // Clear for next event
                roi_rows_seen      <= 20'd0;

                prev_y_all         <= {Y_W{1'b0}};
                prev_y_roi         <= {Y_W{1'b0}};
                have_prev_all      <= 1'b0;
                have_prev_roi      <= 1'b0;

                cur_run_all        <= 8'd0;
                best_run_all       <= 8'd0;

                cur_run_roi        <= 8'd0;
                best_run_roi       <= 8'd0;
            end
        end
    end

endmodule
