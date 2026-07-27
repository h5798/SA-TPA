param(
    [string]$RunName = "ablations_v1"
)

$ErrorActionPreference = "Stop"
$Python = "D:\Anaconda_envs\YOLO\python.exe"
$Project = "D:\456\project"
$FeatureRoot = "D:\456\data\processed\clip_features"
$ResultRoot = "D:\456\results\ablations"
$ResultsCsv = "$ResultRoot\$RunName.csv"
$Methods = @("prompt_ensemble", "source_anchored_text", "no_source_anchor", "satpa_no_uncertainty", "satpa")
$Tasks = @(
    @("A2W", "amazon", "webcam"), @("D2W", "dslr", "webcam"),
    @("W2A", "webcam", "amazon"), @("A2D", "amazon", "dslr"),
    @("D2A", "dslr", "amazon"), @("W2D", "webcam", "dslr")
)

foreach ($Task in $Tasks) {
    foreach ($Method in $Methods) {
        $TaskName = $Task[0]
        $Source = "$FeatureRoot\office31_$($Task[1])_vitb32_openai.npz"
        $Target = "$FeatureRoot\office31_$($Task[2])_vitb32_openai.npz"
        $Prediction = "$ResultRoot\predictions\$RunName\$TaskName`_$Method.npz"
        if (Test-Path -LiteralPath $Prediction) { continue }
        & $Python "$Project\scripts\run_task.py" --run-tag $RunName --task $TaskName --method $Method `
            --source-features $Source --target-features $Target --prediction-output $Prediction `
            --results-csv $ResultsCsv | Out-Null
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    Write-Output "COMPLETE $($Task[0])"
}

