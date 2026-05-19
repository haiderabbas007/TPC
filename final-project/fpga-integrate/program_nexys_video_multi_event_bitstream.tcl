# program_nexys_video_multi_event_bitstream.tcl

open_hw_manager
connect_hw_server -url localhost:3121

puts "============================================================"
puts "Available hardware targets:"
puts [get_hw_targets]
puts "============================================================"

set targets [get_hw_targets]

if {[llength $targets] == 0} {
    puts "ERROR: No hardware targets found."
    puts "Check board power and PROG/JTAG USB cable."
    exit 1
}

set target [lindex $targets 0]
open_hw_target $target

puts "============================================================"
puts "Available hardware devices:"
puts [get_hw_devices]
puts "============================================================"

set devs [get_hw_devices]

if {[llength $devs] == 0} {
    puts "ERROR: No hardware devices found after opening target."
    exit 1
}

set dev [lindex $devs 0]

puts "Programming device:"
puts $dev

current_hw_device $dev
refresh_hw_device $dev

set_property PROGRAM.FILE {C:/hls_stream_medium/hls4ml_medium_stream_cnn_80_180/nexys_video_multi_event_bitstream_output/nexys_video_multi_event_top.bit} $dev

program_hw_devices $dev
refresh_hw_device $dev

puts "============================================================"
puts "Multi-event bitstream programming complete."
puts ""
puts "LED mapping:"
puts "LED0 = MMCM locked"
puts "LED1 = reset released"
puts "LED2 = event slot bit 0"
puts "LED3 = event slot bit 1"
puts "LED4 = result captured"
puts "LED5 = hardware trigger"
puts "LED6 = expected trigger"
puts "LED7 = match indicator"
puts ""
puts "Expected event sequence:"
puts "slot 0: event 001, expected trigger 1"
puts "slot 1: event 011, expected trigger 0"
puts "slot 2: event 012, expected trigger 1"
puts "slot 3: event 013, expected trigger 0"
puts "============================================================"