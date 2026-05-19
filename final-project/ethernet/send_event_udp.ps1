# send_event_udp.ps1
#
# Sends one 2000-sample RTL event file as 4 UDP packets.
#
# Usage example:
# powershell -ExecutionPolicy Bypass -File .\ethernet_stage\send_event_udp.ps1 `
#   -EventFile rtl_events\event_input_001.txt `
#   -EventId 1 `
#   -TargetIp 127.0.0.1 `
#   -TargetPort 5005

param(
    [string]$EventFile = "rtl_events\event_input_001.txt",
    [int]$EventId = 1,
    [string]$TargetIp = "127.0.0.1",
    [int]$TargetPort = 5005
)

$Magic = 0xCAFE
$TotalPackets = 4
$SamplesPerPacket = 500

Write-Host "============================================================"
Write-Host "UDP Event Sender"
Write-Host "Event file:   $EventFile"
Write-Host "Event ID:     $EventId"
Write-Host "Target IP:    $TargetIp"
Write-Host "Target port:  $TargetPort"
Write-Host "============================================================"

$samples = Get-Content $EventFile |
    Where-Object { $_.Trim() -ne "" } |
    ForEach-Object { [int]($_.Trim()) }

if ($samples.Count -ne 2000) {
    throw "Expected 2000 samples, got $($samples.Count)"
}

Write-Host "[+] Loaded samples: $($samples.Count)"
Write-Host "[+] Min sample: $(($samples | Measure-Object -Minimum).Minimum)"
Write-Host "[+] Max sample: $(($samples | Measure-Object -Maximum).Maximum)"

$udp = New-Object System.Net.Sockets.UdpClient

function Add-U16LE {
    param(
        [System.Collections.Generic.List[byte]]$Bytes,
        [int]$Value
    )

    if ($Value -lt 0 -or $Value -gt 65535) {
        throw "Value out of uint16 range: $Value"
    }

    $b = [System.BitConverter]::GetBytes([UInt16]$Value)

    if (-not [System.BitConverter]::IsLittleEndian) {
        [Array]::Reverse($b)
    }

    $Bytes.Add($b[0])
    $Bytes.Add($b[1])
}

for ($pkt = 0; $pkt -lt $TotalPackets; $pkt++) {
    $offset = $pkt * $SamplesPerPacket
    $count  = $SamplesPerPacket

    $bytes = New-Object 'System.Collections.Generic.List[byte]'

    Add-U16LE -Bytes $bytes -Value $Magic
    Add-U16LE -Bytes $bytes -Value $EventId
    Add-U16LE -Bytes $bytes -Value $pkt
    Add-U16LE -Bytes $bytes -Value $TotalPackets
    Add-U16LE -Bytes $bytes -Value $offset
    Add-U16LE -Bytes $bytes -Value $count

    for ($i = 0; $i -lt $count; $i++) {
        $v = $samples[$offset + $i]
        Add-U16LE -Bytes $bytes -Value $v
    }

    $payload = $bytes.ToArray()

    # Debug header check before sending.
    $magic_check = [System.BitConverter]::ToUInt16($payload, 0)
    $offset_check = [System.BitConverter]::ToUInt16($payload, 8)
    $count_check = [System.BitConverter]::ToUInt16($payload, 10)

    $sent = $udp.Send($payload, $payload.Length, $TargetIp, $TargetPort)

    Write-Host ("[+] Sent packet {0}/{1}: magic=0x{2:X4}, offset={3}, samples={4}, bytes={5}" -f `
        ($pkt + 1), $TotalPackets, $magic_check, $offset_check, $count_check, $sent)

    Start-Sleep -Milliseconds 20
}

$udp.Close()

Write-Host "============================================================"
Write-Host "Done."
Write-Host "============================================================"