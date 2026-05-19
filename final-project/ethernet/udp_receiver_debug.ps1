# udp_receiver_debug.ps1
#
# Simple UDP receiver to inspect event packets.
#
# Usage:
# powershell -ExecutionPolicy Bypass -File .\ethernet_stage\udp_receiver_debug.ps1 -ListenPort 5005

param(
    [int]$ListenPort = 5005
)

function Read-U16LE {
    param(
        [byte[]]$Bytes,
        [int]$Index
    )

    return [int][System.BitConverter]::ToUInt16($Bytes, $Index)
}

$udp = New-Object System.Net.Sockets.UdpClient($ListenPort)
$remote = New-Object System.Net.IPEndPoint([System.Net.IPAddress]::Any, 0)

Write-Host "============================================================"
Write-Host "UDP Receiver Debug"
Write-Host "Listening on port $ListenPort"
Write-Host "Press Ctrl+C to stop."
Write-Host "============================================================"

while ($true) {
    $data = $udp.Receive([ref]$remote)

    if ($data.Length -lt 12) {
        Write-Host "Received short packet: $($data.Length) bytes"
        continue
    }

    $magic        = Read-U16LE -Bytes $data -Index 0
    $event_id     = Read-U16LE -Bytes $data -Index 2
    $packet_index = Read-U16LE -Bytes $data -Index 4
    $total_pkts   = Read-U16LE -Bytes $data -Index 6
    $offset       = Read-U16LE -Bytes $data -Index 8
    $count        = Read-U16LE -Bytes $data -Index 10

    Write-Host ("From {0}:{1} | magic=0x{2:X4}, event_id={3}, packet={4}/{5}, offset={6}, count={7}, bytes={8}" -f `
        $remote.Address, $remote.Port, $magic, $event_id, ($packet_index + 1), $total_pkts, $offset, $count, $data.Length)
}