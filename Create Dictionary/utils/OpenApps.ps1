# tls-cycle.ps1
# Formål: Starte apper, trigge nettaktivitet, vente, hard-kille, repetere.
# Logger til CSV (tidsstempel per fase) for enkel korrelasjon mot PCAP/Zeek.

$LogPath = ".\tls_cycle_log.csv"

# Antall runder og tidsparametre
$Iterations      = 30
$WarmupSeconds   = 10      # kort tid rett etter start (la prosessene komme opp)
$ActiveSeconds   = 25      # tid hvor vi aktivt trigget nett + lar appen jobbe
$CooldownSeconds = 5

# Nett-triggere (enkle, men effektive): flere domener gir flere handshakes
$Urls = @(
  "https://www.microsoft.com",
  "https://teams.microsoft.com",
  "https://login.microsoftonline.com",
  "https://www.office.com",
  "https://www.cloudflare.com"
)

# Appliste: tilpass paths/prosessnavn
$Apps = @(
  @{
    Name="Teams"
    # Teams kan hete ms-teams (new) eller Teams (classic). Ta med flere:
    Processes=@("ms-teams","Teams","Microsoft.Teams")
    PathCandidates=@(
      "$env:LOCALAPPDATA\Microsoft\Teams\current\Teams.exe",
      "$env:ProgramFiles\WindowsApps\*Teams*",
      "$env:LOCALAPPDATA\Microsoft\WindowsApps\ms-teams.exe"
    )
    Args=""
  },
  @{
    Name="Outlook"
    Processes=@("OUTLOOK")
    PathCandidates=@(
      "C:\Program Files\Microsoft Office\root\Office16\OUTLOOK.EXE",
      "C:\Program Files (x86)\Microsoft Office\root\Office16\OUTLOOK.EXE"
    )
    Args=""
  },
  @{
    Name="Chrome"
    Processes=@("chrome")
    PathCandidates=@(
      "C:\Program Files\Google\Chrome\Application\chrome.exe",
      "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    )
    # Incognito + nye prosesser gjør det mer sannsynlig med nye connections
    Args="--incognito --new-window https://example.com"
  }
)

function Write-LogRow($phase, $appName, $extra="") {
  $ts = (Get-Date).ToString("o")
  $row = [PSCustomObject]@{
    timestamp = $ts
    phase     = $phase
    app       = $appName
    extra     = $extra
  }
  if (-not (Test-Path $LogPath)) {
    $row | Export-Csv -Path $LogPath -NoTypeInformation
  } else {
    $row | Export-Csv -Path $LogPath -NoTypeInformation -Append
  }
}

function Stop-Processes([string[]]$names) {
  foreach ($n in $names) {
    Get-Process -Name $n -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  }
}

function Resolve-AppPath([string[]]$candidates) {
  foreach ($c in $candidates) {
    # Hvis wildcard (f.eks WindowsApps), prøv å resolve:
    if ($c -like "*`**") {
      $hit = Get-ChildItem -Path $c -ErrorAction SilentlyContinue | Select-Object -First 1
      if ($hit) { return $hit.FullName }
    } elseif (Test-Path $c) {
      return $c
    }
  }
  return $null
}

function Trigger-Network([string[]]$urls) {
  # Ren TLS/DNS-trigging uten browser: BITS/Invoke-WebRequest.
  # Dette er ikke identisk med app-traffic, men gir ekstra handshakes og “støy” om dere vil.
  foreach ($u in $urls | Sort-Object {Get-Random}) {
    try {
      Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 10 | Out-Null
    } catch {
      # ignorer
    }
  }
}

# (Valgfritt) Reduser sjansen for at DNS-cache maskerer ting:
function Flush-Dns {
  try { ipconfig /flushdns | Out-Null } catch {}
}

Write-LogRow "start" "script" "iterations=$Iterations"

for ($i=1; $i -le $Iterations; $i++) {
  foreach ($app in $Apps) {
    $name = $app.Name

    Write-LogRow "iteration_begin" $name "i=$i"

    # 1) Hard kill alt relatert
    Write-LogRow "kill_begin" $name
    Stop-Processes $app.Processes

    # Teams (særlig) kan ha WebView2/Edge prosesser som holder TLS åpent.
    # (Valgfritt) kill WebView2/Edge-bakgrunn for renere “ny” handshake:
    if ($name -eq "Teams") {
      Stop-Processes @("msedgewebview2","msedge","WebViewHost")
    }
    Write-LogRow "kill_end" $name

    Start-Sleep -Seconds 2

    # 2) Finn exe
    $exe = Resolve-AppPath $app.PathCandidates
    if (-not $exe) {
      Write-LogRow "error" $name "exe_not_found"
      continue
    }

    # 3) Start app
    Write-LogRow "launch_begin" $name $exe
    try {
      Start-Process -FilePath $exe -ArgumentList $app.Args | Out-Null
      Write-LogRow "launch_end" $name
    } catch {
      Write-LogRow "error" $name "launch_failed"
      continue
    }

    # 4) Warmup
    Start-Sleep -Seconds $WarmupSeconds

    # 5) Trigger nett (for å garantere TLS-aktivitet også hvis app er “idle”)
    Write-LogRow "net_trigger_begin" $name
    Flush-Dns
    Trigger-Network $Urls
    Write-LogRow "net_trigger_end" $name

    # 6) Aktiv periode
    Start-Sleep -Seconds $ActiveSeconds

    # 7) Hard kill igjen
    Write-LogRow "kill2_begin" $name
    Stop-Processes $app.Processes
    if ($name -eq "Teams") {
      Stop-Processes @("msedgewebview2","msedge","WebViewHost")
    }
    Write-LogRow "kill2_end" $name

    Start-Sleep -Seconds $CooldownSeconds
    Write-LogRow "iteration_end" $name "i=$i"
  }
}

Write-LogRow "end" "script" ""
Write-Host "Done. Logg: $LogPath"
