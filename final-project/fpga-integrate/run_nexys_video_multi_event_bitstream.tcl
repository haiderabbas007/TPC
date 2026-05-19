# run_nexys_video_multi_event_bitstream.tcl

open_project vivado_nexys_video_multi_event_project/nexys_video_multi_event_project.xpr

set_property top nexys_video_multi_event_top [get_filesets sources_1]
update_compile_order -fileset sources_1

reset_run synth_1
reset_run impl_1

launch_runs synth_1 -jobs 4
wait_on_run synth_1

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

open_run impl_1

file mkdir nexys_video_multi_event_bitstream_output

set bit_files [glob -nocomplain vivado_nexys_video_multi_event_project/nexys_video_multi_event_project.runs/impl_1/*.bit]
foreach f $bit_files {
    file copy -force $f nexys_video_multi_event_bitstream_output/
}

report_timing_summary -file nexys_video_multi_event_bitstream_output/nexys_video_multi_event_timing_summary.rpt
report_utilization -file nexys_video_multi_event_bitstream_output/nexys_video_multi_event_utilization.rpt
report_power -file nexys_video_multi_event_bitstream_output/nexys_video_multi_event_power.rpt
report_route_status -file nexys_video_multi_event_bitstream_output/nexys_video_multi_event_route_status.rpt

puts "============================================================"
puts "Nexys Video multi-event bitstream build finished."
puts "Check folder:"
puts "nexys_video_multi_event_bitstream_output"
puts "Expected: .bit file plus reports"
puts "============================================================"