`timescale 1ns / 1ps

module axis_cnn_dma_wrapper (
    input  wire        ap_clk,
    input  wire        ap_rst_n,

    // AXI Stream input from AXI DMA MM2S
    input  wire [15:0] s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire        s_axis_tlast,

    // AXI Stream output to AXI DMA S2MM
    output wire [31:0] m_axis_tdata,
    output wire [3:0]  m_axis_tkeep,
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,
    output wire        m_axis_tlast
);

    wire ap_done;
    wire ap_idle;
    wire ap_ready;

    // For this test, keep the HLS block enabled.
    // The CNN consumes one event stream and eventually produces one result word.
    wire ap_start;
    assign ap_start = 1'b1;

    // All 4 bytes of the 32-bit result word are valid.
    assign m_axis_tkeep = 4'hF;

    // The CNN wrapper produces exactly one result word per event.
    // So that result word is also the last word of the output packet.
    assign m_axis_tlast = m_axis_tvalid;

    // Existing validated wrapper
    cnn_trigger_axis_result_wrapper u_cnn_result (
        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),

        .ap_start(ap_start),
        .ap_done(ap_done),
        .ap_idle(ap_idle),
        .ap_ready(ap_ready),

        .input_layer_V_TDATA(s_axis_tdata),
        .input_layer_V_TVALID(s_axis_tvalid),
        .input_layer_V_TREADY(s_axis_tready),

        .result_TDATA(m_axis_tdata),
        .result_TVALID(m_axis_tvalid),
        .result_TREADY(m_axis_tready)
    );

    // s_axis_tlast is intentionally not used here.
    // The HLS CNN knows the event length from its internal design.
    // The DMA still sends TLAST after the input event, but the CNN wrapper
    // does not need it.

endmodule
