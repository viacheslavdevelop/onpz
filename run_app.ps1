param(
    [switch]$NoBrowser,
    [int]$Port = 7860,
    [int]$LlamaPort = 8080,
    [string]$LlamaServer = "",
    [string]$Model = "",
    [switch]$NoLlama,
    [switch]$Share,
    [string]$ShareAuth = "",
    [switch]$Check
)

$env:PYTHONIOENCODING = 'utf-8'
Set-Location $PSScriptRoot

$arguments = @(
    'run_app.py',
    '--ui-port', "$Port",
    '--llama-port', "$LlamaPort"
)

if ($NoBrowser) { $arguments += '--no-browser' }
if ($LlamaServer) { $arguments += @('--llama-server', $LlamaServer) }
if ($Model) { $arguments += @('--model', $Model) }
if ($NoLlama) { $arguments += '--no-llama' }
if ($Share) { $arguments += '--share' }
if ($ShareAuth) { $arguments += @('--share-auth', $ShareAuth) }
if ($Check) { $arguments += '--check' }

python @arguments
exit $LASTEXITCODE
