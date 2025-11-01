Param(
  [ValidateSet('cpu','gpu','all','bundle')][string]$Profile='cpu',
  [ValidateSet('build','up','down','restart','logs','bundle')][string]$Action='up',
  [string]$Output='offline-bundle',
  [switch]$CpuOnly,
  [switch]$GpuOnly,
  [switch]$NoZip,
  [switch]$AddHash
)

$ComposeFile = 'cluster/gpu-cluster.yml'

function Build-ApiGateway {
  Write-Host 'Building api-gateway image...'
  docker compose -f $ComposeFile build api-gateway
}

function Up-Profile($profile) {
  if ($profile -eq 'all') {
    docker compose -f $ComposeFile --profile cpu --profile gpu up -d --build
  } elseif ($profile -eq 'bundle') {
    Write-Host 'Bundle profile used only with -Action bundle'
  } else {
    docker compose -f $ComposeFile --profile $profile up -d --build
  }
}

function Down-All { docker compose -f $ComposeFile down }
function Restart-Profile($profile){ docker compose -f $ComposeFile --profile $profile up -d --build }
function Show-Logs { docker compose -f $ComposeFile logs --tail=100 }

function Bundle-Images {
  Write-Host 'Preparing image list...'
  $baseImages = @('local/api-gateway:latest')
  $cpuImages = @('ollama/ollama:latest','rayproject/ray:latest-py311','nginx:alpine')
  $gpuImages = @('vllm/vllm-openai:latest')

  if ($CpuOnly -and $GpuOnly) { throw 'Cannot set both -CpuOnly and -GpuOnly.' }

  $images = @()
  $images += $baseImages
  if ($GpuOnly) { $images += $gpuImages } elseif ($CpuOnly) { $images += $cpuImages } else { $images += $cpuImages + $gpuImages }

  Write-Host 'Pulling upstream images...'
  ($images | Where-Object { $_ -ne 'local/api-gateway:latest' }) | ForEach-Object { docker pull $_ }
  Write-Host 'Building local image api-gateway...'
  docker compose -f $ComposeFile build api-gateway

  $tarFile = "$Output.tar"
  Write-Host "Saving images ($($images.Count)) to $tarFile ..."
  docker save -o $tarFile $images

  if ($AddHash) {
    $hash = (Get-FileHash -Algorithm SHA256 $tarFile).Hash
    Set-Content -Path "$Output.sha256" -Value $hash
    Write-Host "Wrote SHA256 hash to $Output.sha256"
  }

  if (-not $NoZip) {
    Write-Host 'Compressing bundle (zip)...'
    Compress-Archive -Path $tarFile -DestinationPath "$Output.zip" -Force
    if ($AddHash) {
      $zipHash = (Get-FileHash -Algorithm SHA256 "$Output.zip").Hash
      Set-Content -Path "$Output.zip.sha256" -Value $zipHash
      Write-Host "Wrote ZIP SHA256 hash to $Output.zip.sha256"
    }
    Write-Host "Created $Output.zip"
    Write-Host 'Transfer ZIP via USB. On target:'
    Write-Host "  Expand-Archive -Path $Output.zip -DestinationPath ."
  } else {
    Write-Host 'Skipping zip compression as requested (-NoZip). Transfer TAR directly.'
  }

  Write-Host "Load images offline: docker load -i $tarFile"
  if ($CpuOnly) { Write-Host 'Start CPU profile: docker compose -f cluster/gpu-cluster.yml --profile cpu up -d' }
  elseif ($GpuOnly) { Write-Host 'Start GPU profile: docker compose -f cluster/gpu-cluster.yml --profile gpu up -d' }
  else { Write-Host 'Start all profiles: docker compose -f cluster/gpu-cluster.yml --profile cpu --profile gpu up -d' }
}

switch ($Action) {
  'build' { Build-ApiGateway }
  'up'    { Up-Profile $Profile }
  'down'  { Down-All }
  'restart' { Restart-Profile $Profile }
  'logs' { Show-Logs }
  'bundle' { Bundle-Images }
}
