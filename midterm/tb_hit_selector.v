`timescale 1ns/1ps

module tb_hit_selector;

    localparam integer X_W = 5;
    localparam integer Y_W = 8;

    localparam integer X_MIN = 0;
    localparam integer X_MAX = 19;
    localparam integer Y_MIN = 120;
    localparam integer Y_MAX = 150;

    localparam integer DISPLAY_HITS = 0;
    localparam integer THRESHOLD    = 5;

    // 0=train, 1=val, 2=test
    localparam integer SPLIT_SEL    = 0;

    reg                  clk;
    reg                  rst;
    reg                  valid_in;
    reg  [X_W-1:0]       x_in;
    reg  [Y_W-1:0]       y_in;
    wire                 valid_out;
    wire                 label_out;

    integer list_file, event_file, r;
    integer x_val, y_val;
    integer true_label;
    integer pred_label;
    integer hit_count;
    integer roi_hit_count;
    integer event_count;
    integer TP, TN, FP, FN;
    integer listed_label;

    reg [8*256-1:0] event_name;
    reg [8*512-1:0] event_path;
    reg [8*512-1:0] list_path;

    hit_selector #(
        .X_W(X_W), .Y_W(Y_W),
        .X_MIN(X_MIN), .X_MAX(X_MAX),
        .Y_MIN(Y_MIN), .Y_MAX(Y_MAX)
    ) dut (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in),
        .x_in(x_in),
        .y_in(y_in),
        .valid_out(valid_out),
        .label_out(label_out)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst         = 1'b1;
        valid_in    = 1'b0;
        x_in        = 0;
        y_in        = 0;
        event_count = 0;
        TP = 0; TN = 0; FP = 0; FN = 0;

        #20;
        rst = 1'b0;

        // Build filelist path based on split selection
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

        $display("=== Starting dataset evaluation ===");
        $display("ROI: x in [%0d,%0d], y in [%0d,%0d]", X_MIN, X_MAX, Y_MIN, Y_MAX);
        $display("Threshold: roi_hit_count >= %0d => predict 1", THRESHOLD);
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
                    event_count   = event_count + 1;
                    true_label    = listed_label;
                    pred_label    = 0;
                    hit_count     = 0;
                    roi_hit_count = 0;

                    if (DISPLAY_HITS)
                        $display("\n=== Event %0d: %0s ===", event_count, event_name);

                    while (!$feof(event_file)) begin
                        r = $fscanf(event_file, "%d %d\n", x_val, y_val);

                        if (r == 2) begin
                            hit_count = hit_count + 1;

                            @(posedge clk);
                            valid_in <= 1'b1;
                            x_in     <= x_val[X_W-1:0];
                            y_in     <= y_val[Y_W-1:0];

                            @(posedge clk);
                            #1;

                            if (valid_out && label_out)
                                roi_hit_count = roi_hit_count + 1;

                            if (DISPLAY_HITS) begin
                                $display("  hit %0d: x=%0d y=%0d -> valid_out=%0b label_out=%0b",
                                         hit_count, x_val, y_val, valid_out, label_out);
                            end

                            valid_in <= 1'b0;
                            x_in     <= 0;
                            y_in     <= 0;
                        end
                    end

                    @(posedge clk);
                    #1;
                    if (valid_out && label_out)
                        roi_hit_count = roi_hit_count + 1;

                    $fclose(event_file);

                    if (roi_hit_count >= THRESHOLD)
                        pred_label = 1;
                    else
                        pred_label = 0;

                    if (true_label == 1 && pred_label == 1) TP = TP + 1;
                    else if (true_label == 0 && pred_label == 0) TN = TN + 1;
                    else if (true_label == 0 && pred_label == 1) FP = FP + 1;
                    else if (true_label == 1 && pred_label == 0) FN = FN + 1;

                    $display("Event %0d: true=%0d pred=%0d hits=%0d roi_hits=%0d | TP=%0d TN=%0d FP=%0d FN=%0d",
                             event_count, true_label, pred_label, hit_count, roi_hit_count,
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
