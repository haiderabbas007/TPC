# make_multi_event_rom_4.ps1
# Generates multi_event_rom_4.v from four real exported RTL event files.

$events = @(
    @{ slot = 0; index = 1;  file = "rtl_events/event_input_001.txt"; expected = 1 },
    @{ slot = 1; index = 11; file = "rtl_events/event_input_011.txt"; expected = 0 },
    @{ slot = 2; index = 12; file = "rtl_events/event_input_012.txt"; expected = 1 },
    @{ slot = 3; index = 13; file = "rtl_events/event_input_013.txt"; expected = 0 }
)

$outfile = "multi_event_rom_4.v"

$all_values = @{}

foreach ($e in $events) {
    $vals = Get-Content $e.file |
        Where-Object { $_.Trim() -ne "" } |
        ForEach-Object { [int]($_.Trim()) }

    if ($vals.Count -ne 2000) {
        throw "Expected 2000 samples in $($e.file), got $($vals.Count)"
    }

    $all_values[$e.slot] = $vals

    Write-Host "[+] Slot $($e.slot): event $($e.index), expected trigger $($e.expected), file $($e.file)"
    Write-Host "    samples=$($vals.Count), min=$(($vals | Measure-Object -Minimum).Minimum), max=$(($vals | Measure-Object -Maximum).Maximum)"
}

$sw = New-Object System.IO.StreamWriter($outfile, $false)

$sw.WriteLine('`timescale 1ns / 1ps')
$sw.WriteLine('')
$sw.WriteLine('module multi_event_rom_4 (')
$sw.WriteLine('    input  wire [1:0]  event_sel,')
$sw.WriteLine('    input  wire [11:0] addr,')
$sw.WriteLine('    output reg  [15:0] data,')
$sw.WriteLine('    output reg         expected_trigger')
$sw.WriteLine(');')
$sw.WriteLine('')
$sw.WriteLine('    always @(*) begin')
$sw.WriteLine("        data = 16'd0;")
$sw.WriteLine("        expected_trigger = 1'b0;")
$sw.WriteLine('        case (event_sel)')

foreach ($e in $events) {
    $slot = $e.slot
    $expected = $e.expected
    $vals = $all_values[$slot]

    $sw.WriteLine("            2'd$($slot): begin")
    $sw.WriteLine("                expected_trigger = 1'b$($expected);")
    $sw.WriteLine('                case (addr)')

    for ($i = 0; $i -lt $vals.Count; $i++) {
        $sw.WriteLine("                    12'd$($i): data = 16'd$($vals[$i]);")
    }

    $sw.WriteLine("                    default: data = 16'd0;")
    $sw.WriteLine('                endcase')
    $sw.WriteLine('            end')
}

$sw.WriteLine('            default: begin')
$sw.WriteLine("                data = 16'd0;")
$sw.WriteLine("                expected_trigger = 1'b0;")
$sw.WriteLine('            end')
$sw.WriteLine('        endcase')
$sw.WriteLine('    end')
$sw.WriteLine('')
$sw.WriteLine('endmodule')

$sw.Close()

Write-Host "[+] Wrote $outfile"