param(
    [string]$OutputPath = "sysmon_data.csv",
    [int]$PollSeconds = 1,
    [int]$BootstrapMaxEvents = 100 # Number of historical events to load on startup if the output file doesn't exist
)

$logName = 'Microsoft-Windows-Sysmon/Operational'
$eventId = 3

function Convert-SysmonEventToObject {
    param([Parameter(Mandatory = $true)]$Event)

    [PSCustomObject]@{
        timestamp = $Event.TimeCreated.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ")
        app       = $Event.Properties[4].Value
        srcport   = $Event.Properties[11].Value
        dstport   = $Event.Properties[16].Value
        dst       = $Event.Properties[14].Value
        domain    = $Event.Properties[15].Value
    }
}

Write-Host "[*] Starting live Sysmon stream (Event ID $eventId). Press Ctrl+C to stop."
Write-Host "[*] Writing to: $OutputPath"

$existingRecordId = $null
$latestExistingEvent = Get-WinEvent -FilterHashtable @{ LogName = $logName; ID = $eventId } -MaxEvents 1 -ErrorAction SilentlyContinue
if ($latestExistingEvent) {
    $existingRecordId = $latestExistingEvent.RecordId
}

if (-not (Test-Path -Path $OutputPath)) {
    Write-Host "[*] Bootstrapping with the last $BootstrapMaxEvents historical events..."
    $bootstrapEvents = Get-WinEvent -FilterHashtable @{ LogName = $logName; ID = $eventId } -MaxEvents $BootstrapMaxEvents -ErrorAction SilentlyContinue |
        Sort-Object RecordId

    if ($bootstrapEvents) {
        $bootstrapRows = $bootstrapEvents | ForEach-Object { Convert-SysmonEventToObject -Event $_ }
        $bootstrapRows | Export-Csv -Path $OutputPath -NoTypeInformation
        $existingRecordId = ($bootstrapEvents | Select-Object -Last 1).RecordId
        Write-Host "[+] Bootstrapped $($bootstrapRows.Count) events."
    }
    else {
        Write-Host "[*] No historical events found for bootstrap."
    }
}

while ($true) {
    $events = Get-WinEvent -FilterHashtable @{ LogName = $logName; ID = $eventId } -MaxEvents 200 -ErrorAction SilentlyContinue |
        Sort-Object RecordId

    if ($events) {
        $newEvents = if ($null -eq $existingRecordId) {
            $events
        }
        else {
            $events | Where-Object { $_.RecordId -gt $existingRecordId }
        }

        if ($newEvents) {
            $rows = $newEvents | ForEach-Object { Convert-SysmonEventToObject -Event $_ }
            $rows | Export-Csv -Path $OutputPath -NoTypeInformation -Append

            foreach ($row in $rows) {
                Write-Host ("[{0}] srcport={1} -> {2}:{3} | app={4} | domain={5}" -f $row.timestamp, $row.srcport, $row.dst, $row.dstport, $row.app, $row.domain)
            }

            $existingRecordId = ($newEvents | Select-Object -Last 1).RecordId
        }
    }

    Start-Sleep -Seconds $PollSeconds
}