$body = @{
    text = "Bitcoin is going to the moon!"
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/predict-dev" -Method POST -ContentType "application/json" -Body $body -UseBasicParsing
    Write-Host "Status Code: $($response.StatusCode)"
    Write-Host "Response: $($response.Content)"
} catch {
    Write-Host "Error: $($_.Exception.Message)"
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response Body: $responseBody"
    }
}
