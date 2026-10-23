$files = Get-ChildItem 'C:\My\University\onpz\Processed data\*.md'
$total = 0
foreach ($f in $files) {
    $c = Get-Content $f.FullName -Raw
    $words = ($c -split '\s+').Count
    $chars = $c.Length
    Write-Output ("{0}: {1} chars, {2} words" -f $f.Name, $chars, $words)
    $total += $chars
}
Write-Output ("TOTAL CHARS: {0}" -f $total)
