<#
run_judge.ps1

Runs the judge_engine using the repository-local virtual environment (relative paths).

Usage:
  PowerShell (from repository root):
    .\run_judge.ps1 [args...]

This script will try to use `.venv\Scripts\python.exe`. If not found, it falls back to `python` on PATH.
#>

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    $RemainingArgs
)

# Determine the script directory (repo root when executed from repo root)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Relative paths (from repository root)
$venvPython = Join-Path $scriptDir '.venv\Scripts\python.exe'
$judgeScript = Join-Path $scriptDir 'case_closed\case-closed-starter-code-main\case-closed-starter-code-main\judge_engine.py'

if (-not (Test-Path $judgeScript)) {
    Write-Error "Could not find judge script at relative path: $judgeScript"
    exit 2
}

if (Test-Path $venvPython) {
    Write-Host "Using virtual env python: $venvPython"
    & $venvPython $judgeScript @RemainingArgs
    exit $LASTEXITCODE
} else {
    Write-Host "Virtual env python not found at $venvPython. Falling back to 'python' on PATH."
    & python $judgeScript @RemainingArgs
    exit $LASTEXITCODE
}
