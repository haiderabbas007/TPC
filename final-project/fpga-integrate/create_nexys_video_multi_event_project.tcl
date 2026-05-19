# create_nexys_video_multi_event_project.tcl

set proj_name "nexys_video_multi_event_project"
set proj_dir  "vivado_nexys_video_multi_event_project"

create_project $proj_name $proj_dir -part xc7a200tsbg484-1 -force

add_files -norecurse [glob ./myproject_prj/solution1/syn/verilog/*.v]

add_files -norecurse ./axis_simple_fifo.v
add_files -norecurse ./cnn_trigger_wrapper.v
add_files -norecurse ./cnn_trigger_axis_result_wrapper.v
add_files -norecurse ./cnn_trigger_fifo_top.v
add_files -norecurse ./multi_event_rom_4.v
add_files -norecurse ./nexys_video_multi_event_top.v

add_files -fileset constrs_1 -norecurse ./nexys_video_multi_event_top.xdc

set_property top nexys_video_multi_event_top [get_filesets sources_1]
update_compile_order -fileset sources_1

puts "============================================================"
puts "Created Nexys Video multi-event ROM project."
puts "Top:  nexys_video_multi_event_top"
puts "Part: xc7a200tsbg484-1"
puts "Slots:"
puts "0 -> event_input_001, expected trigger 1"
puts "1 -> event_input_011, expected trigger 0"
puts "2 -> event_input_012, expected trigger 1"
puts "3 -> event_input_013, expected trigger 0"
puts "============================================================"