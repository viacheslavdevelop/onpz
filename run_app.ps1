param(
    [switch]$NoBrowser,
    [int]$Port = 7860
)

$env:PYTHONIOENCODING = 'utf-8'
Set-Location $PSScriptRoot

$arguments = @('App/app.py', '--port', $Port)
if ($NoBrowser) {
    $arguments += '--no-browser'
}

python @arguments
exit $LASTEXITCODE
