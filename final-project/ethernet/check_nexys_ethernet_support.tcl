# check_nexys_ethernet_support.tcl

create_project check_nexys_ethernet_support ./vivado_check_nexys_ethernet_support -part xc7a200tsbg484-1 -force

puts "============================================================"
puts "Checking gmii_to_rgmii IP..."
puts "============================================================"

set ips [get_ipdefs -all *gmii*]
foreach ip $ips {
    puts $ip
}

puts "============================================================"
puts "Checking MicroBlaze IP..."
puts "============================================================"

set mb_ips [get_ipdefs -all *microblaze*]
foreach ip $mb_ips {
    puts $ip
}

puts "============================================================"
puts "Checking AXI Timer / UART / GPIO / BRAM IP..."
puts "============================================================"

foreach pattern {"*axi_timer*" "*axi_uartlite*" "*axi_gpio*" "*axi_bram_ctrl*" "*blk_mem_gen*" "*proc_sys_reset*" "*axi_intc*"} {
    puts "---- $pattern ----"
    set found [get_ipdefs -all $pattern]
    foreach ip $found {
        puts $ip
    }
}

puts "============================================================"
puts "Checking installed board parts containing nexys..."
puts "============================================================"

set boards [get_board_parts *nexys*]
foreach b $boards {
    puts $b
}

puts "============================================================"
puts "Done."
puts "============================================================"

close_project