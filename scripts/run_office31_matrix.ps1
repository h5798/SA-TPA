param(
    [string]$RunName = "development_v1"
)

$ErrorActionPreference = "Stop"
$Python = "D:\Anaconda_envs\YOLO\python.exe"
$Project = "D:\456\project"
$FeatureRoot = "D:\456\data\processed\clip_features"
$ResultRoot = "D:\456\results\office31"
$ResultsCsv = "$ResultRoot\$RunName.csv"
$Methods = @("clip_zero_shot", "prompt_ensemble", "source_prototype", "no_source_anchor", "satpa")
$Tasks = @(
    @("A2W", "amazon", "webcam"),
    @("D2W", "dslr", "webcam"),
    @("W2A", "webcam", "amazon"),
    @("A2D", "amazon", "dslr"),
    @("D2A", "dslr", "amazon"),
    @("W2D", "webcam", "dslr")
)

foreach ($Task in $Tasks) {
    $TaskName = $Task[0]
    $Source = "$FeatureRoot\office31_$($Task[1])_vitb32_openai.npz"
    $Target = "$FeatureRoot\office31_$($Task[2])_vitb32_openai.npz"
    foreach ($Method in $Methods) {
        $Prediction = "$ResultRoot\predictions\$RunName\$TaskName`_$Method.npz"
        if (Test-Path -LiteralPath $Prediction) {
            Write-Output "SKIP $TaskName $Method (prediction exists)"
            continue
        }
        & $Python "$Project\scripts\run_task.py" `
            --task $TaskName `
            --method $Method `
            --source-features $Source `
            --target-features $Target `
            --prediction-output $Prediction `
            --results-csv $ResultsCsv
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

