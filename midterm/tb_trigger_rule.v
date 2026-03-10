`timescale 1ns / 1ps

module tb_trigger_rule;

    localparam integer ROI_ACTIVE_ROWS_W   = 4;
    localparam integer LONGEST_ROW_RUN_W   = 8;
    localparam integer ROI_LONGEST_Y_RUN_W = 8;
    localparam integer LONGEST_Y_RUN_W     = 8;

    reg  [ROI_ACTIVE_ROWS_W-1:0]   roi_active_rows;
    reg  [LONGEST_ROW_RUN_W-1:0]   longest_row_run;
    reg  [ROI_LONGEST_Y_RUN_W-1:0] roi_longest_y_run;
    reg  [LONGEST_Y_RUN_W-1:0]     longest_y_run;

    wire trigger_out;
    wire branch1_out;
    wire branch2_out;

    trigger_rule #(
        .ROI_ACTIVE_ROWS_W(ROI_ACTIVE_ROWS_W),
        .LONGEST_ROW_RUN_W(LONGEST_ROW_RUN_W),
        .ROI_LONGEST_Y_RUN_W(ROI_LONGEST_Y_RUN_W),
        .LONGEST_Y_RUN_W(LONGEST_Y_RUN_W)
    ) dut (
        .roi_active_rows(roi_active_rows),
        .longest_row_run(longest_row_run),
        .roi_longest_y_run(roi_longest_y_run),
        .longest_y_run(longest_y_run),
        .trigger_out(trigger_out),
        .branch1_out(branch1_out),
        .branch2_out(branch2_out)
    );

    task run_case;
        input [ROI_ACTIVE_ROWS_W-1:0]   t_roi_active_rows;
        input [LONGEST_ROW_RUN_W-1:0]   t_longest_row_run;
        input [ROI_LONGEST_Y_RUN_W-1:0] t_roi_longest_y_run;
        input [LONGEST_Y_RUN_W-1:0]     t_longest_y_run;
        begin
            roi_active_rows   = t_roi_active_rows;
            longest_row_run   = t_longest_row_run;
            roi_longest_y_run = t_roi_longest_y_run;
            longest_y_run     = t_longest_y_run;

            #1;
            $display("roi_active_rows=%0d longest_row_run=%0d roi_longest_y_run=%0d longest_y_run=%0d | branch1=%0b branch2=%0b trigger=%0b",
                     roi_active_rows, longest_row_run, roi_longest_y_run, longest_y_run,
                     branch1_out, branch2_out, trigger_out);
            #9;
        end
    endtask

    initial begin
        $display("=== trigger_rule unit test ===");

        // Branch 1 should fire
        run_case(2, 2, 2, 5);

        // Branch 2 should fire
        run_case(4, 1, 1, 10);

        // Both should fire
        run_case(2, 5, 6, 12);

        // Neither should fire
        run_case(5, 1, 1, 7);

        // Neither should fire
        run_case(3, 1, 1, 8);

        // Branch 2 only
        run_case(4, 0, 0, 15);

        // Branch 1 only
        run_case(1, 3, 4, 2);

        $display("=== end of trigger_rule unit test ===");
        $finish;
    end

endmodule
