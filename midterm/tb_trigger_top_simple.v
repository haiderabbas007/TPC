`timescale 1ns/1ps

module tb_trigger_top_simple;

    localparam integer X_W = 5;
    localparam integer Y_W = 8;

    localparam integer X_MIN = 0;
    localparam integer X_MAX = 19;
    localparam integer Y_MIN = 120;
    localparam integer Y_MAX = 150;

    // 0=train, 1=val, 2=test
    localparam integer SPLIT_SEL = 2;

    reg                  clk;
    reg                  rst;
    reg                  valid_in;
    reg                  event_done;
    reg  [X_W-1:0]       x_in;
    reg  [Y_W-1:0]       y_in;

    wire                 valid_out;
    wire                 label_out;

    wire [3:0]           roi_active_rows_dbg;
    wire [7:0]           longest_row_run_dbg;
    wire [7:0]           roi_longest_y_run_dbg;
    wire [7:0]           longest_y_run_dbg;
    wire                 branch1_dbg;
    wire                 branch2_dbg;

    integer list_file, event_file, r;
    integer x_val, y_val;
    integer true_label;
    integer pred_label;
    integer hit_count;
    integer event_count;
    integer TP, TN, FP, FN;
    integer listed_label;

    reg [8*256-1:0] event_name;
    reg [8*512-1:0] event_path;
    reg [8*512-1:0] list_path;

    trigger_top_simple #(
        .X_W(X_W), .Y_W(Y_W),
        .X_MIN(X_MIN), .X_MAX(X_MAX),
        .Y_MIN(Y_MIN), .Y_MAX(Y_MAX)
    ) dut (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in),
        .x_in(x_in),
        .y_in(y_in),
        .event_done(event_done),
        .valid_out(valid_out),
        .label_out(label_out),
        .roi_active_rows_dbg(roi_active_rows_dbg),
        .longest_row_run_dbg(longest_row_run_dbg),
        .roi_longest_y_run_dbg(roi_longest_y_run_dbg),
        .longest_y_run_dbg(longest_y_run_dbg),
        .branch1_dbg(branch1_dbg),
        .branch2_dbg(branch2_dbg)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst         = 1'b1;
        valid_in    = 1'b0;
        event_done  = 1'b0;
        x_in        = 0;
        y_in        = 0;
        event_count = 0;
        TP = 0; TN = 0; FP = 0; FN = 0;

        #20;
        rst = 1'b0;

        if (SPLIT_SEL == 0)
            $sformat(list_path,
                "C:/Users/Haider Abbas/Downloads/PHYS 476/Midterm/fpga_input/train/filelist.txt");
        else if (SPLIT_SEL == 1)
            $sformat(list_path,
                "C:/Users/Haider Abbas/Downloads/PHYS 476/Midterm/fpga_input/val/filelist.txt");
        else
            $sformat(list_path,
                "C:/Users/Haider Abbas/Downloads/PHYS 476/Midterm/fpga_input/test/filelist.txt");

        list_file = $fopen(list_path, "r");

        if (list_file == 0) begin
            $display("ERROR: could not open %0s", list_path);
            $finish;
        end

        $display("=== Starting trigger_top_simple dataset evaluation ===");
        $display("Using list: %0s", list_path);

        while (!$feof(list_file)) begin
            event_name   = 0;
            listed_label = 0;

            r = $fscanf(list_file, "%s %d\n", event_name, listed_label);

            if (r == 2) begin
                if (SPLIT_SEL == 0)
                    $sformat(event_path,
                        "C:/Users/Haider Abbas/Downloads/PHYS 476/Midterm/fpga_input/train/%0s",
                        event_name);
                else if (SPLIT_SEL == 1)
                    $sformat(event_path,
                        "C:/Users/Haider Abbas/Downloads/PHYS 476/Midterm/fpga_input/val/%0s",
                        event_name);
                else
                    $sformat(event_path,
                        "C:/Users/Haider Abbas/Downloads/PHYS 476/Midterm/fpga_input/test/%0s",
                        event_name);

                event_file = $fopen(event_path, "r");

                if (event_file == 0) begin
                    $display("WARNING: could not open %0s", event_path);
                end else begin
                    event_count = event_count + 1;
                    true_label  = listed_label;
                    pred_label  = 0;
                    hit_count   = 0;

                    while (!$feof(event_file)) begin
                        r = $fscanf(event_file, "%d %d\n", x_val, y_val);

                        if (r == 2) begin
                            hit_count = hit_count + 1;

                            @(posedge clk);
                            valid_in   <= 1'b1;
                            event_done <= 1'b0;
                            x_in       <= x_val[X_W-1:0];
                            y_in       <= y_val[Y_W-1:0];

                            @(posedge clk);
                            valid_in   <= 1'b0;
                            event_done <= 1'b0;
                            x_in       <= 0;
                            y_in       <= 0;
                        end
                    end

                    $fclose(event_file);

                    // End-of-event pulse
                    @(posedge clk);
                    valid_in   <= 1'b0;
                    event_done <= 1'b1;
                    x_in       <= 0;
                    y_in       <= 0;

                    @(posedge clk);
                    event_done <= 1'b0;

                    #1;
                    if (valid_out)
                        pred_label = label_out;
                    else
                        pred_label = 0;

                    if (true_label == 1 && pred_label == 1) TP = TP + 1;
                    else if (true_label == 0 && pred_label == 0) TN = TN + 1;
                    else if (true_label == 0 && pred_label == 1) FP = FP + 1;
                    else if (true_label == 1 && pred_label == 0) FN = FN + 1;

                    $display("Event %0d: true=%0d pred=%0d hits=%0d | roi_active_rows=%0d longest_row_run=%0d roi_longest_y_run=%0d longest_y_run=%0d | b1=%0b b2=%0b | TP=%0d TN=%0d FP=%0d FN=%0d",
                             event_count, true_label, pred_label, hit_count,
                             roi_active_rows_dbg, longest_row_run_dbg, roi_longest_y_run_dbg, longest_y_run_dbg,
                             branch1_dbg, branch2_dbg,
                             TP, TN, FP, FN);

                    repeat (3) @(posedge clk);
                end
            end
        end

        $fclose(list_file);

        $display("");
        $display("=== FINAL RESULTS ===");
        $display("Events processed = %0d", event_count);
        $display("TP = %0d", TP);
        $display("TN = %0d", TN);
        $display("FP = %0d", FP);
        $display("FN = %0d", FN);

        if (event_count > 0)
            $display("Accuracy = %0f", (1.0 * (TP + TN)) / event_count);

        repeat (10) @(posedge clk);
        $finish;
    end

endmodule
