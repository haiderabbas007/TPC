# check_ethernet_ip_available.tcl

create_project check_ethernet_ip_available ./vivado_check_ethernet_ip_available -part xc7a200tsbg484-1 -force

puts "============================================================"
puts "Searching available Ethernet-related IP cores..."
puts "============================================================"

set ips [get_ipdefs -all *ethernet*]
foreach ip $ips {
    puts $ip
}

puts "============================================================"
puts "Searching AXI-related Ethernet IP cores..."
puts "============================================================"

set ips2 [get_ipdefs -all *axi*ethernet*]
foreach ip $ips2 {
    puts $ip
}

puts "============================================================"
puts "Searching tri-mode Ethernet MAC IP cores..."
puts "============================================================"

set ips3 [get_ipdefs -all *tri*mode*]
foreach ip $ips3 {
    puts $ip
}

puts "============================================================"
puts "Done."
puts "============================================================"

close_project