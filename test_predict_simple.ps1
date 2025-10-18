$ErrorActionPreference = "Stop"

$body = @{
    text = "Bitcoin is going to the moon!"
} | ConvertTo-Json

Write-Host "Testing predict-dev endpoint..."
Write-Host "Request body: $body"

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/predict-dev" -Method POST -ContentType "application/json" -Body $body -UseBasicParsing
    Write-Host "Status Code: $($response.StatusCode)"
    Write-Host "Response: $($response.Content)"
} catch {
    Write-Host "Error: $($_.Exception.Message)"
    Write-Host "Status Code: $($_.Exception.Response.StatusCode.value__)"
    
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response Body: $responseBody"
        $reader.Close()
    }
}

