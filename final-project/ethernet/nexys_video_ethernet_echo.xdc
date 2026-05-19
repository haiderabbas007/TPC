# nexys_video_ethernet_echo.xdc
# Manual constraints for Nexys Video Ethernet echo block design.
#
# Top wrapper ports confirmed:
#   CLK100MHZ
#   mdio_0_mdc
#   mdio_0_mdio_io
#   phy_rst_n_0[0]
#   reset_rtl_0
#   rgmii_0_rd[3:0]
#   rgmii_0_rx_ctl
#   rgmii_0_rxc
#   rgmii_0_td[3:0]
#   rgmii_0_tx_ctl
#   rgmii_0_txc
#   uart_rtl_0_rxd
#   uart_rtl_0_txd

# ============================================================
# 100 MHz board clock
# ============================================================

set_property -dict { PACKAGE_PIN R4 IOSTANDARD LVCMOS33 } [get_ports { CLK100MHZ }]
create_clock -period 10.000 -name CLK100MHZ -waveform {0.000 5.000} [get_ports { CLK100MHZ }]

# ============================================================
# CPU reset button
# ============================================================

set_property -dict { PACKAGE_PIN G4 IOSTANDARD LVCMOS15 } [get_ports { reset_rtl_0 }]

# ============================================================
# USB UART bridge
# ============================================================

set_property -dict { PACKAGE_PIN AA19 IOSTANDARD LVCMOS33 } [get_ports { uart_rtl_0_rxd }]
set_property -dict { PACKAGE_PIN V18  IOSTANDARD LVCMOS33 } [get_ports { uart_rtl_0_txd }]

# ============================================================
# Ethernet PHY: Realtek RTL8211E RGMII + MDIO
#
# Corrected Nexys Video RGMII mapping:
#
# MDC       -> AA16
# MDIO      -> Y16
# PHY reset -> U7
#
# RXC       -> V13
# RXCTL     -> W10
# RXD0      -> AB16
# RXD1      -> AA15
# RXD2      -> AB15
# RXD3      -> AB11
#
# TXC       -> AA14
# TXCTL     -> V10
# TXD0      -> Y12
# TXD1      -> W12
# TXD2      -> W11
# TXD3      -> Y11
# ============================================================

# MDIO / MDC
set_property -dict { PACKAGE_PIN AA16 IOSTANDARD LVCMOS25 } [get_ports { mdio_0_mdc }]
set_property -dict { PACKAGE_PIN Y16  IOSTANDARD LVCMOS25 } [get_ports { mdio_0_mdio_io }]

# Ethernet PHY reset, active-low
set_property -dict { PACKAGE_PIN U7 IOSTANDARD LVCMOS33 } [get_ports { phy_rst_n_0[0] }]

# RGMII receive side: PHY -> FPGA
set_property -dict { PACKAGE_PIN V13  IOSTANDARD LVCMOS25 } [get_ports { rgmii_0_rxc }]
set_property -dict { PACKAGE_PIN W10  IOSTANDARD LVCMOS25 } [get_ports { rgmii_0_rx_ctl }]
set_property -dict { PACKAGE_PIN AB16 IOSTANDARD LVCMOS25 } [get_ports { rgmii_0_rd[0] }]
set_property -dict { PACKAGE_PIN AA15 IOSTANDARD LVCMOS25 } [get_ports { rgmii_0_rd[1] }]
set_property -dict { PACKAGE_PIN AB15 IOSTANDARD LVCMOS25 } [get_ports { rgmii_0_rd[2] }]
set_property -dict { PACKAGE_PIN AB11 IOSTANDARD LVCMOS25 } [get_ports { rgmii_0_rd[3] }]

# RGMII transmit side: FPGA -> PHY
set_property -dict { PACKAGE_PIN AA14 IOSTANDARD LVCMOS25 } [get_ports { rgmii_0_txc }]
set_property -dict { PACKAGE_PIN V10  IOSTANDARD LVCMOS25 } [get_ports { rgmii_0_tx_ctl }]
set_property -dict { PACKAGE_PIN Y12  IOSTANDARD LVCMOS25 } [get_ports { rgmii_0_td[0] }]
set_property -dict { PACKAGE_PIN W12  IOSTANDARD LVCMOS25 } [get_ports { rgmii_0_td[1] }]
set_property -dict { PACKAGE_PIN W11  IOSTANDARD LVCMOS25 } [get_ports { rgmii_0_td[2] }]
set_property -dict { PACKAGE_PIN Y11  IOSTANDARD LVCMOS25 } [get_ports { rgmii_0_td[3] }]

# Output tuning for RGMII transmit signals
set_property SLEW FAST [get_ports { rgmii_0_td[*] }]
set_property SLEW FAST [get_ports { rgmii_0_tx_ctl }]
set_property SLEW FAST [get_ports { rgmii_0_txc }]

# ============================================================
# Notes
# ============================================================
#
# Do not paste this file into the Tcl Console.
# Save this as the .xdc constraint file.
#
# The earlier placement failure came from wrong Ethernet pin/IO-standard
# assignments. RGMII pins are constrained here using the corrected Nexys
# Video Ethernet mapping and LVCMOS25 for the RGMII/MDIO bank.