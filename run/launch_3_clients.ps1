# Launch 3 Minecraft clients for a single-worker MARL run
# Each instance is configured with reduced memory to fit on a single machine.

# run : powershell -NoProfile -ExecutionPolicy Bypass -File .\run\launch_3_clients.ps1

$MALMO_PATH = "C:\Malmo\Minecraft"
$env:MALMO_XSD_PATH = "C:\Malmo\Schemas"

# Reduce heap size per instance.
# 512MB is usually enough for these training environments.
$env:_JAVA_OPTIONS = "-Xmx512m -Xms256m -XX:+UseG1GC"

Write-Host "Starting 3 Minecraft instances..." -ForegroundColor Cyan

for ($i = 0; $i -lt 3; $i++) {
    $port = 10000 + $i
    Write-Host "Launching instance on port $port..."

    # Start each client in its own cmd window so you can monitor them
    $cmdArgs = "/k cd /d `"$MALMO_PATH`" && gradlew.bat runClient -Pport=$port"
    Start-Process -FilePath "cmd.exe" -ArgumentList $cmdArgs

    # Stagger launches to avoid CPU spikes during startup
    if ($i -lt 2) {
        Start-Sleep -Seconds 60
    }
}

Write-Host "Done. Wait for all instances to reach the Main Menu before starting training." -ForegroundColor Green