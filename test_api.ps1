$headers = @{
    'Authorization' = 'Bearer dev-key-123'
    'Content-Type' = 'application/json'
}

$body = @{
    text = "Bitcoin is going to the moon!"
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/predict" -Method POST -Headers $headers -Body $body -UseBasicParsing
    Write-Host "Status Code: $($response.StatusCode)"
    Write-Host "Response: $($response.Content)"
} catch {
    Write-Host "Error: $($_.Exception.Message)"
    Write-Host "Response: $($_.Exception.Response)"
}
