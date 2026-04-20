# Launch 6 Minecraft clients for MARL training
# Each instance is configured with reduced memory to fit on a single machine.

$MALMO_PATH = "C:\Malmo\Minecraft"
$env:MALMO_XSD_PATH = "C:\Malmo\Schemas"

# Reduce heap size per instance. 
# 512MB is usually enough for these training environments.
$env:_JAVA_OPTIONS = "-Xmx512m -Xms256m -XX:+UseG1GC"

Write-Host "Starting 6 Minecraft instances..." -ForegroundColor Cyan

for ($i = 0; $i -lt 6; $i++) {
    $port = 10000 + $i
    Write-Host "Launching instance on port $port..."
    
    # Start each client in a new window so you can monitor them
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd $MALMO_PATH; .\gradlew.bat runClient -Pport=$port"
    
    # Stagger launches to avoid CPU spikes during startup
    if ($i -lt 5) {
        Start-Sleep -Seconds 5
    }
}

Write-Host "Done. Wait for all instances to reach the Main Menu before starting training." -ForegroundColor Green
