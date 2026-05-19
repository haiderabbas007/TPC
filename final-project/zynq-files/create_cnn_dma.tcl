# ============================================================
# PYNQ-Z2 CNN + AXI DMA overlay
#
# Python DDR event buffer
#   -> AXI DMA MM2S, 16-bit stream
#   -> axis_cnn_dma_wrapper
#   -> AXI DMA S2MM, 32-bit result stream
#   -> Python DDR result buffer
#
# Exports:
#   C:/pynq_cnn_dma/export/cnn_dma.bit
#   C:/pynq_cnn_dma/export/cnn_dma.hwh
# ============================================================

set proj_name "pynq_z2_cnn_dma"
set bd_name   "cnn_dma_bd"

set origin_dir [pwd]
set proj_dir "$origin_dir/$proj_name"

set hls_rtl_dir "C:/hls_stream_medium/hls4ml_medium_stream_cnn_80_180/myproject_prj/solution1/impl/verilog"
set wrap_dir    "C:/hls_stream_medium/hls4ml_medium_stream_cnn_80_180/final_validated_axis_trigger_block"

puts "Creating project in $proj_dir"

create_project $proj_name $proj_dir -part xc7z020clg400-1 -force

# Use PYNQ-Z2 board files if available
set board_parts [get_board_parts -quiet *pynq-z2*]
if {[llength $board_parts] > 0} {
    set_property board_part [lindex $board_parts 0] [current_project]
    puts "Using board part: [lindex $board_parts 0]"
} else {
    puts "WARNING: PYNQ-Z2 board part not found. Using raw part xc7z020clg400-1."
}

# ------------------------------------------------------------
# Add RTL sources
# ------------------------------------------------------------
set hls_files [glob -nocomplain "$hls_rtl_dir/*.v"]

if {[llength $hls_files] == 0} {
    puts "ERROR: No HLS Verilog files found in $hls_rtl_dir"
    exit 1
}

add_files -norecurse $hls_files
add_files -norecurse "$wrap_dir/cnn_trigger_wrapper.v"
add_files -norecurse "$wrap_dir/cnn_trigger_axis_result_wrapper.v"
add_files -norecurse "$origin_dir/axis_cnn_dma_wrapper.v"

update_compile_order -fileset sources_1

# ------------------------------------------------------------
# Create block design
# ------------------------------------------------------------
create_bd_design $bd_name
current_bd_design $bd_name

# ------------------------------------------------------------
# Zynq PS
# ------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 processing_system7_0

if {[llength $board_parts] > 0} {
    apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
        -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
        [get_bd_cells processing_system7_0]
}

set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100.0} \
] [get_bd_cells processing_system7_0]

# ------------------------------------------------------------
# AXI DMA
# ------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma axi_dma_0

set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} \
    CONFIG.c_m_axis_mm2s_tdata_width {16} \
    CONFIG.c_s_axis_s2mm_tdata_width {32} \
    CONFIG.c_m_axi_mm2s_data_width {32} \
    CONFIG.c_m_axi_s2mm_data_width {32} \
    CONFIG.c_addr_width {32} \
] [get_bd_cells axi_dma_0]

# ------------------------------------------------------------
# CNN wrapper RTL module
# ------------------------------------------------------------
create_bd_cell -type module -reference axis_cnn_dma_wrapper axis_cnn_dma_wrapper_0

# ------------------------------------------------------------
# Reset block
# ------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_ps7_0_100M

# ------------------------------------------------------------
# AXI control interconnect:
# PS M_AXI_GP0 -> DMA S_AXI_LITE
# ------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect axi_ctrl_interconnect_0
set_property -dict [list \
    CONFIG.NUM_SI {1} \
    CONFIG.NUM_MI {1} \
] [get_bd_cells axi_ctrl_interconnect_0]

# ------------------------------------------------------------
# AXI memory interconnect:
# DMA MM2S/S2MM memory ports -> PS HP0 DDR port
# ------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect axi_mem_interconnect_0
set_property -dict [list \
    CONFIG.NUM_SI {2} \
    CONFIG.NUM_MI {1} \
] [get_bd_cells axi_mem_interconnect_0]

# ------------------------------------------------------------
# Clocks
# ------------------------------------------------------------
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins rst_ps7_0_100M/slowest_sync_clk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_100M/ext_reset_in]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_dma_0/s_axi_lite_aclk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axis_cnn_dma_wrapper_0/ap_clk]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_ctrl_interconnect_0/ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_ctrl_interconnect_0/S00_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_ctrl_interconnect_0/M00_ACLK]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_mem_interconnect_0/ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_mem_interconnect_0/S00_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_mem_interconnect_0/S01_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_mem_interconnect_0/M00_ACLK]

# ------------------------------------------------------------
# Resets
# ------------------------------------------------------------
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] [get_bd_pins axi_dma_0/axi_resetn]
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] [get_bd_pins axis_cnn_dma_wrapper_0/ap_rst_n]

connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] [get_bd_pins axi_ctrl_interconnect_0/ARESETN]
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] [get_bd_pins axi_ctrl_interconnect_0/S00_ARESETN]
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] [get_bd_pins axi_ctrl_interconnect_0/M00_ARESETN]

connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] [get_bd_pins axi_mem_interconnect_0/ARESETN]
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] [get_bd_pins axi_mem_interconnect_0/S00_ARESETN]
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] [get_bd_pins axi_mem_interconnect_0/S01_ARESETN]
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] [get_bd_pins axi_mem_interconnect_0/M00_ARESETN]

# ------------------------------------------------------------
# AXI-Lite control path
# ------------------------------------------------------------
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins axi_ctrl_interconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_ctrl_interconnect_0/M00_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

# ------------------------------------------------------------
# AXI memory path
# ------------------------------------------------------------
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_interconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_interconnect_0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_mem_interconnect_0/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

# ------------------------------------------------------------
# AXI Stream connections, pin-by-pin
# DMA MM2S -> CNN input
# CNN output -> DMA S2MM
# ------------------------------------------------------------

# Input stream: DMA to CNN
connect_bd_net [get_bd_pins axi_dma_0/m_axis_mm2s_tdata]  [get_bd_pins axis_cnn_dma_wrapper_0/s_axis_tdata]
connect_bd_net [get_bd_pins axi_dma_0/m_axis_mm2s_tvalid] [get_bd_pins axis_cnn_dma_wrapper_0/s_axis_tvalid]
connect_bd_net [get_bd_pins axi_dma_0/m_axis_mm2s_tready] [get_bd_pins axis_cnn_dma_wrapper_0/s_axis_tready]
connect_bd_net [get_bd_pins axi_dma_0/m_axis_mm2s_tlast]  [get_bd_pins axis_cnn_dma_wrapper_0/s_axis_tlast]

# Output stream: CNN to DMA
connect_bd_net [get_bd_pins axis_cnn_dma_wrapper_0/m_axis_tdata]  [get_bd_pins axi_dma_0/s_axis_s2mm_tdata]
connect_bd_net [get_bd_pins axis_cnn_dma_wrapper_0/m_axis_tkeep]  [get_bd_pins axi_dma_0/s_axis_s2mm_tkeep]
connect_bd_net [get_bd_pins axis_cnn_dma_wrapper_0/m_axis_tvalid] [get_bd_pins axi_dma_0/s_axis_s2mm_tvalid]
connect_bd_net [get_bd_pins axis_cnn_dma_wrapper_0/m_axis_tready] [get_bd_pins axi_dma_0/s_axis_s2mm_tready]
connect_bd_net [get_bd_pins axis_cnn_dma_wrapper_0/m_axis_tlast]  [get_bd_pins axi_dma_0/s_axis_s2mm_tlast]

# ------------------------------------------------------------
# Address assignment
# ------------------------------------------------------------
assign_bd_address

# ------------------------------------------------------------
# Validate/save
# ------------------------------------------------------------
validate_bd_design
save_bd_design

# ------------------------------------------------------------
# Wrapper
# ------------------------------------------------------------
set bd_file [get_files "$proj_dir/$proj_name.srcs/sources_1/bd/$bd_name/$bd_name.bd"]
make_wrapper -files $bd_file -top

set wrapper_file "$proj_dir/$proj_name.gen/sources_1/bd/$bd_name/hdl/${bd_name}_wrapper.v"
add_files -norecurse $wrapper_file
set_property top ${bd_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

# ------------------------------------------------------------
# Build
# ------------------------------------------------------------
puts "Launching synthesis..."
launch_runs synth_1 -jobs 4
wait_on_run synth_1
puts "Synthesis status: [get_property STATUS [get_runs synth_1]]"

puts "Launching implementation + bitstream..."
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
puts "Implementation status: [get_property STATUS [get_runs impl_1]]"

# ------------------------------------------------------------
# Export overlay files
# ------------------------------------------------------------
set export_dir "$origin_dir/export"
file mkdir $export_dir

set bit_file "$proj_dir/$proj_name.runs/impl_1/${bd_name}_wrapper.bit"
set hwh_file "$proj_dir/$proj_name.gen/sources_1/bd/$bd_name/hw_handoff/${bd_name}.hwh"

if {[file exists $bit_file]} {
    file copy -force $bit_file "$export_dir/cnn_dma.bit"
    puts "Copied bit file to $export_dir/cnn_dma.bit"
} else {
    puts "ERROR: bit file not found at $bit_file"
}

if {[file exists $hwh_file]} {
    file copy -force $hwh_file "$export_dir/cnn_dma.hwh"
    puts "Copied hwh file to $export_dir/cnn_dma.hwh"
} else {
    puts "ERROR: hwh file not found at $hwh_file"
}

puts "DONE. Export folder:"
puts $export_dir
