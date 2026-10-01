# Enables Baton's versioned git hooks for this clone (Windows / PowerShell).
# Run once after cloning:   .\scripts\install-hooks.ps1
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "git was not found on PATH. Install Git for Windows, or run this from a shell where git is available."
    }
    git config core.hooksPath .githooks
    if ($LASTEXITCODE -ne 0) { throw "git config failed." }
    Write-Host "Git hooks enabled (core.hooksPath = .githooks)."
    Write-Host "Every commit is now scanned for secrets by scripts/secret_scan.py."
}
finally {
    Pop-Location
}
