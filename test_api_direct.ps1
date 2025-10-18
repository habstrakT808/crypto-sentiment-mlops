# Test API directly
$body = @{
    text = "Bitcoin is going to the moon!"
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/predict-dev" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing
    Write-Host "Status: $($response.StatusCode)"
    Write-Host "Response: $($response.Content)"
} catch {
    Write-Host "Error: $($_.Exception.Message)"
}
