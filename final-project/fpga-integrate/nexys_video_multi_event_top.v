`timescale 1ns / 1ps

module nexys_video_multi_event_top (
    input  wire        CLK100MHZ,
    input  wire        cpu_resetn,
    input  wire        btnc,
    output wire [7:0]  led
);

    // ============================================================
    // 100 MHz board clock -> 80 MHz CNN clock
    // ============================================================

    wire clkfb;
    wire clk80_unbuf;
    wire clk80;
    wire mmcm_locked;

    MMCME2_BASE #(
        .BANDWIDTH("OPTIMIZED"),
        .CLKFBOUT_MULT_F(8.000),
        .CLKFBOUT_PHASE(0.000),
        .DIVCLK_DIVIDE(1),
        .CLKIN1_PERIOD(10.000),
        .CLKOUT0_DIVIDE_F(10.000),
        .CLKOUT0_PHASE(0.000),
        .CLKOUT0_DUTY_CYCLE(0.500),
        .STARTUP_WAIT("FALSE")
    ) mmcm_80mhz_inst (
        .CLKIN1(CLK100MHZ),
        .CLKFBIN(clkfb),
        .CLKFBOUT(clkfb),
        .CLKOUT0(clk80_unbuf),
        .LOCKED(mmcm_locked),
        .PWRDWN(1'b0),
        .RST(~cpu_resetn)
    );

    BUFG clk80_bufg_inst (
        .I(clk80_unbuf),
        .O(clk80)
    );

    // ============================================================
    // Reset synchronizer
    // ============================================================

    reg [7:0] reset_shift = 8'h00;

    always @(posedge clk80 or negedge cpu_resetn) begin
        if (!cpu_resetn) begin
            reset_shift <= 8'h00;
        end else if (!mmcm_locked) begin
            reset_shift <= 8'h00;
        end else begin
            reset_shift <= {reset_shift[6:0], 1'b1};
        end
    end

    wire ap_rst_n;
    assign ap_rst_n = reset_shift[7];

    // ============================================================
    // Raw button synchronizer
    // ============================================================

    reg btnc_meta = 1'b0;
    reg btnc_sync = 1'b0;

    always @(posedge clk80) begin
        if (!ap_rst_n) begin
            btnc_meta <= 1'b0;
            btnc_sync <= 1'b0;
        end else begin
            btnc_meta <= btnc;
            btnc_sync <= btnc_meta;
        end
    end

    // ============================================================
    // Button debouncer / one-pulse generator
    // ============================================================
    //
    // clk80 = 80 MHz.
    // 20 ms debounce time = 1,600,000 cycles.
    //
    // This block waits until the synchronized button input has been
    // stable for about 20 ms, then updates the debounced state.
    //
    // btnc_clean_rise is a single-cycle pulse generated only when the
    // debounced button changes from 0 to 1.
    //
    // This prevents one physical press from being counted multiple times.
    // ============================================================

    localparam integer DEBOUNCE_COUNT_MAX = 1600000;
    localparam integer DEBOUNCE_COUNT_W   = 21;

    reg [DEBOUNCE_COUNT_W-1:0] debounce_count = {DEBOUNCE_COUNT_W{1'b0}};
    reg btnc_last_sample = 1'b0;
    reg btnc_debounced   = 1'b0;
    reg btnc_debounced_d = 1'b0;

    always @(posedge clk80) begin
        if (!ap_rst_n) begin
            debounce_count <= {DEBOUNCE_COUNT_W{1'b0}};
            btnc_last_sample <= 1'b0;
            btnc_debounced <= 1'b0;
            btnc_debounced_d <= 1'b0;
        end else begin
            btnc_debounced_d <= btnc_debounced;

            if (btnc_sync != btnc_last_sample) begin
                btnc_last_sample <= btnc_sync;
                debounce_count <= {DEBOUNCE_COUNT_W{1'b0}};
            end else begin
                if (debounce_count < DEBOUNCE_COUNT_MAX[DEBOUNCE_COUNT_W-1:0]) begin
                    debounce_count <= debounce_count + {{(DEBOUNCE_COUNT_W-1){1'b0}}, 1'b1};
                end else begin
                    btnc_debounced <= btnc_last_sample;
                end
            end
        end
    end

    wire btnc_clean_rise;
    assign btnc_clean_rise = btnc_debounced & ~btnc_debounced_d;

    // ============================================================
    // Expected trigger lookup from event slot
    // ============================================================
    //
    // slot 0 -> event_input_001, expected trigger 1
    // slot 1 -> event_input_011, expected trigger 0
    // slot 2 -> event_input_012, expected trigger 1
    // slot 3 -> event_input_013, expected trigger 0
    // ============================================================

    function expected_for_slot;
        input [1:0] sel;
        begin
            case (sel)
                2'd0: expected_for_slot = 1'b1;
                2'd1: expected_for_slot = 1'b0;
                2'd2: expected_for_slot = 1'b1;
                2'd3: expected_for_slot = 1'b0;
                default: expected_for_slot = 1'b0;
            endcase
        end
    endfunction

    // ============================================================
    // Multi-event ROM
    // ============================================================

    reg  [1:0]  current_sel;
    reg  [1:0]  run_sel;
    reg  [1:0]  captured_sel;

    reg  [11:0] rom_addr;
    wire [15:0] rom_data;
    wire        rom_expected_trigger_unused;

    multi_event_rom_4 event_rom (
        .event_sel(run_sel),
        .addr(rom_addr),
        .data(rom_data),
        .expected_trigger(rom_expected_trigger_unused)
    );

    reg captured_expected_trigger;

    // ============================================================
    // CNN FIFO top connections
    // ============================================================

    reg         ap_start_reg;
    wire        ap_done;
    wire        ap_idle;
    wire        ap_ready;

    wire [15:0] s_axis_TDATA;
    reg         s_axis_TVALID;
    wire        s_axis_TREADY;

    wire [31:0] m_axis_TDATA;
    wire        m_axis_TVALID;
    wire        m_axis_TREADY;

    assign s_axis_TDATA = rom_data;
    assign m_axis_TREADY = 1'b1;

    cnn_trigger_fifo_top cnn_block (
        .ap_clk(clk80),
        .ap_rst_n(ap_rst_n),

        .ap_start(ap_start_reg),
        .ap_done(ap_done),
        .ap_idle(ap_idle),
        .ap_ready(ap_ready),

        .s_axis_TDATA(s_axis_TDATA),
        .s_axis_TVALID(s_axis_TVALID),
        .s_axis_TREADY(s_axis_TREADY),

        .m_axis_TDATA(m_axis_TDATA),
        .m_axis_TVALID(m_axis_TVALID),
        .m_axis_TREADY(m_axis_TREADY)
    );

    // ============================================================
    // Multi-event streamer
    // ============================================================
    //
    // Clean debounced button sequence:
    //
    // Press 1 -> slot 0 -> event_input_001 -> expected trigger 1
    // Press 2 -> slot 1 -> event_input_011 -> expected trigger 0
    // Press 3 -> slot 2 -> event_input_012 -> expected trigger 1
    // Press 4 -> slot 3 -> event_input_013 -> expected trigger 0
    // Press 5 -> slot 0 again
    // ============================================================

    localparam [2:0] ST_IDLE        = 3'd0;
    localparam [2:0] ST_STREAM      = 3'd1;
    localparam [2:0] ST_WAIT_RESULT = 3'd2;
    localparam [2:0] ST_DONE        = 3'd3;

    reg [2:0]  state;
    reg [12:0] sample_count;

    reg [31:0] captured_result;
    reg        result_seen;

    wire captured_trigger;
    assign captured_trigger = captured_result[31];

    wire trigger_match;
    assign trigger_match = result_seen && (captured_trigger == captured_expected_trigger);

    always @(posedge clk80) begin
        if (!ap_rst_n) begin
            state                     <= ST_IDLE;
            sample_count              <= 13'd0;
            rom_addr                  <= 12'd0;

            current_sel               <= 2'd0;
            run_sel                   <= 2'd0;
            captured_sel              <= 2'd0;

            ap_start_reg              <= 1'b0;
            s_axis_TVALID             <= 1'b0;

            captured_result           <= 32'd0;
            result_seen               <= 1'b0;
            captured_expected_trigger <= 1'b0;
        end else begin
            case (state)

                ST_IDLE: begin
                    ap_start_reg  <= 1'b0;
                    s_axis_TVALID <= 1'b0;
                    sample_count  <= 13'd0;
                    rom_addr      <= 12'd0;

                    if (btnc_clean_rise) begin
                        run_sel                   <= current_sel;
                        captured_sel              <= current_sel;
                        captured_expected_trigger <= expected_for_slot(current_sel);

                        captured_result           <= 32'd0;
                        result_seen               <= 1'b0;

                        sample_count              <= 13'd0;
                        rom_addr                  <= 12'd0;

                        ap_start_reg              <= 1'b1;
                        s_axis_TVALID             <= 1'b1;
                        state                     <= ST_STREAM;
                    end
                end

                ST_STREAM: begin
                    ap_start_reg  <= 1'b1;
                    s_axis_TVALID <= 1'b1;

                    if (s_axis_TVALID && s_axis_TREADY) begin
                        if (sample_count == 13'd1999) begin
                            s_axis_TVALID <= 1'b0;
                            sample_count  <= 13'd0;
                            rom_addr      <= 12'd0;
                            state         <= ST_WAIT_RESULT;
                        end else begin
                            sample_count <= sample_count + 13'd1;
                            rom_addr     <= rom_addr + 12'd1;
                        end
                    end
                end

                ST_WAIT_RESULT: begin
                    ap_start_reg  <= 1'b1;
                    s_axis_TVALID <= 1'b0;

                    if (m_axis_TVALID) begin
                        captured_result <= m_axis_TDATA;
                        result_seen     <= 1'b1;
                        ap_start_reg    <= 1'b0;

                        current_sel     <= current_sel + 2'd1;

                        state           <= ST_DONE;
                    end
                end

                ST_DONE: begin
                    ap_start_reg  <= 1'b0;
                    s_axis_TVALID <= 1'b0;

                    if (btnc_clean_rise) begin
                        run_sel                   <= current_sel;
                        captured_sel              <= current_sel;
                        captured_expected_trigger <= expected_for_slot(current_sel);

                        captured_result           <= 32'd0;
                        result_seen               <= 1'b0;

                        sample_count              <= 13'd0;
                        rom_addr                  <= 12'd0;

                        ap_start_reg              <= 1'b1;
                        s_axis_TVALID             <= 1'b1;
                        state                     <= ST_STREAM;
                    end
                end

                default: begin
                    state <= ST_IDLE;
                end

            endcase
        end
    end

    // ============================================================
    // LED mapping
    // ============================================================
    //
    // LED0 = MMCM locked
    // LED1 = reset released
    // LED2 = event slot bit 0
    // LED3 = event slot bit 1
    // LED4 = result seen
    // LED5 = hardware trigger bit
    // LED6 = expected trigger bit
    // LED7 = trigger match indicator
    // ============================================================

    wire [1:0] display_sel;
    assign display_sel = result_seen ? captured_sel : current_sel;

    assign led[0] = mmcm_locked;
    assign led[1] = ap_rst_n;
    assign led[2] = display_sel[0];
    assign led[3] = display_sel[1];
    assign led[4] = result_seen;
    assign led[5] = captured_trigger;
    assign led[6] = captured_expected_trigger;
    assign led[7] = trigger_match;

endmodule