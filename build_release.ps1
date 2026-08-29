$ErrorActionPreference = "Stop"

function Assert-FreeSpace {
  param(
    [Parameter(Mandatory = $true)]
    [string] $Path,
    [Parameter(Mandatory = $true)]
    [long] $RequiredBytes,
    [Parameter(Mandatory = $true)]
    [string] $Operation
  )

  $resolved = [System.IO.Path]::GetFullPath($Path)
  $root = [System.IO.Path]::GetPathRoot($resolved)
  $drive = [System.IO.DriveInfo]::new($root)
  if ($drive.AvailableFreeSpace -lt $RequiredBytes) {
    throw "Insufficient disk for ${Operation}: need $RequiredBytes bytes, have $($drive.AvailableFreeSpace) bytes"
  }
}

function Remove-OldInstallers {
  $bundleRoot = Join-Path $PSScriptRoot "src-tauri\target\release\bundle"
  if (-not (Test-Path -LiteralPath $bundleRoot)) {
    return
  }
  $installers = Get-ChildItem -LiteralPath $bundleRoot -Recurse -File |
    Where-Object { $_.Extension -in @(".exe", ".msi") } |
    Sort-Object LastWriteTimeUtc -Descending
  $installers | Select-Object -Skip 3 | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Force
  }
}

function Invoke-Native {
  param(
    [Parameter(Mandatory = $true)]
    [string] $FilePath,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
  )

  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "$FilePath exited with code $LASTEXITCODE"
  }
}

function Compress-TerminalDb {
  $source = Resolve-Path (Join-Path $PSScriptRoot "..\data\terminal.db")
  $bundleDir = Join-Path $PSScriptRoot "bundle-data"
  $target = Join-Path $bundleDir "terminal.db.gz"
  $staged = Join-Path $bundleDir "terminal.db.gz.staged"
  New-Item -ItemType Directory -Force -Path $bundleDir | Out-Null
  $sourceSize = [long](Get-Item -LiteralPath $source.Path).Length
  Assert-FreeSpace -Path $bundleDir -RequiredBytes ($sourceSize * 2 + 1073741824) -Operation "database compression"

  $inputStream = [System.IO.File]::OpenRead($source)
  $outputStream = [System.IO.File]::Create($staged)
  $gzip = [System.IO.Compression.GZipStream]::new(
    $outputStream,
    [System.IO.Compression.CompressionLevel]::Optimal
  )
  try {
    $inputStream.CopyTo($gzip)
    Write-Host "Compressed terminal.db to: $target"
  } finally {
    $gzip.Dispose()
    $outputStream.Dispose()
    $inputStream.Dispose()
  }
  Move-Item -LiteralPath $staged -Destination $target -Force
}

$previousUvProjectEnvironment = $env:UV_PROJECT_ENVIRONMENT
$previousUvCacheDir = $env:UV_CACHE_DIR
$previousReleasePhase = $env:BHARATTERMINAL_RELEASE_PHASE
Push-Location $PSScriptRoot
try {
  Assert-FreeSpace -Path $PSScriptRoot -RequiredBytes 6442450944 -Operation "release build"
  $releaseEnvironment = Join-Path $PSScriptRoot "compute\.venv-p17-release"
  $env:UV_PROJECT_ENVIRONMENT = $releaseEnvironment
  $env:UV_CACHE_DIR = Join-Path $PSScriptRoot "compute\.uv-cache-p17-release"
  Invoke-Native "uv" sync --project compute --frozen --all-groups
  $releasePython = Join-Path $releaseEnvironment "Scripts\python.exe"
  Invoke-Native $releasePython -m pip check
  Push-Location (Join-Path $PSScriptRoot "compute")
  try {
    Invoke-Native $releasePython -m pytest -c pytest.ini tests -q
  } finally {
    Pop-Location
  }
  Invoke-Native $releasePython scripts\release_gate.py source-openapi
  Compress-TerminalDb
  Invoke-Native $releasePython scripts\release_gate.py generate-identity
  Invoke-Native $releasePython compute\build_compute.py
  Invoke-Native $releasePython scripts\release_gate.py generate-probe-manifest
  Invoke-Native $releasePython scripts\release_gate.py packaged-openapi
  Invoke-Native "corepack" pnpm build
  Invoke-Native $releasePython scripts\release_gate.py generate-prebundle-manifest
  $env:BHARATTERMINAL_RELEASE_PHASE = "PREBUNDLE"
  Invoke-Native "cargo" tauri build --no-bundle
  $env:BHARATTERMINAL_RELEASE_PHASE = $previousReleasePhase
  # `cargo tauri bundle` patches indian-terminal.exe IN PLACE (changes 3
  # bytes: a `__TAURI_BUNDLE_TYPE_VAR_UNK` placeholder becomes `..._NSS` to
  # record the bundle format -- see that command's own "Patching ... with
  # bundle type information: nsis" log line). scripts/release_gate.py's
  # Tauri-executable hashing normalizes that one known marker back to a
  # canonical value before hashing (see sha256_tauri_executable /
  # src-tauri/src/release_integrity.rs's matching normalization), so the
  # manifest hash generated here is stable whether this runs before or
  # after that patch -- no second bundle pass needed.
  Invoke-Native $releasePython scripts\release_gate.py generate-manifest
  Invoke-Native $releasePython scripts\release_gate.py validate
  Invoke-Native "cargo" tauri bundle --bundles nsis
  Invoke-Native $releasePython scripts\release_gate.py validate
  Remove-OldInstallers
  Write-Host "Installer at: src-tauri/target/release/bundle/nsis/"
} finally {
  $env:UV_PROJECT_ENVIRONMENT = $previousUvProjectEnvironment
  $env:UV_CACHE_DIR = $previousUvCacheDir
  $env:BHARATTERMINAL_RELEASE_PHASE = $previousReleasePhase
  Pop-Location
}
