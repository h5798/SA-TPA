param(
    [string]$RunName = "sensitivity_v1"
)

$ErrorActionPreference = "Stop"
$Python = "D:\Anaconda_envs\YOLO\python.exe"
$Project = "D:\456\project"
$FeatureRoot = "D:\456\data\processed\clip_features"
$ResultRoot = "D:\456\results\sensitivity"
$ResultsCsv = "$ResultRoot\$RunName.csv"
$Tasks = @(
    @("A2W", "amazon", "webcam"),
    @("D2W", "dslr", "webcam"),
    @("W2A", "webcam", "amazon"),
    @("A2D", "amazon", "dslr"),
    @("D2A", "dslr", "amazon"),
    @("W2D", "webcam", "dslr")
)

$Configs = @()
foreach ($Value in @(0.0, 0.05, 0.1, 0.2, 0.3)) {
    $Configs += [pscustomobject]@{Tag="alpha_source_$Value"; AlphaSource=$Value; AlphaTarget=0.025; Threshold=0.7; TopK=1; Prior=0.1}
}
foreach ($Value in @(0.0, 0.01, 0.025, 0.05, 0.1)) {
    $Configs += [pscustomobject]@{Tag="alpha_target_$Value"; AlphaSource=0.1; AlphaTarget=$Value; Threshold=0.7; TopK=1; Prior=0.1}
}
foreach ($Value in @(0.5, 0.6, 0.7, 0.8, 0.9)) {
    $Configs += [pscustomobject]@{Tag="threshold_$Value"; AlphaSource=0.1; AlphaTarget=0.025; Threshold=$Value; TopK=1; Prior=0.1}
}
foreach ($Value in @(1, 2, 4, 8)) {
    $Configs += [pscustomobject]@{Tag="top_k_$Value"; AlphaSource=0.1; AlphaTarget=0.025; Threshold=0.7; TopK=$Value; Prior=0.1}
}
foreach ($Value in @(0.0, 0.05, 0.1, 0.2)) {
    $Configs += [pscustomobject]@{Tag="prior_$Value"; AlphaSource=0.1; AlphaTarget=0.025; Threshold=0.7; TopK=1; Prior=$Value}
}

foreach ($Config in $Configs) {
    foreach ($Task in $Tasks) {
        $TaskName = $Task[0]
        $Source = "$FeatureRoot\office31_$($Task[1])_vitb32_openai.npz"
        $Target = "$FeatureRoot\office31_$($Task[2])_vitb32_openai.npz"
        $SafeTag = $Config.Tag.Replace(".", "p")
        $Prediction = "$ResultRoot\predictions\$RunName\$SafeTag`_$TaskName.npz"
        if (Test-Path -LiteralPath $Prediction) { continue }
        & $Python "$Project\scripts\run_task.py" `
            --run-tag $Config.Tag `
            --task $TaskName `
            --method satpa `
            --source-features $Source `
            --target-features $Target `
            --prediction-output $Prediction `
            --results-csv $ResultsCsv `
            --alpha-source $Config.AlphaSource `
            --alpha-target $Config.AlphaTarget `
            --confidence-threshold $Config.Threshold `
            --top-k $Config.TopK `
            --class-prior-strength $Config.Prior | Out-Null
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    Write-Output "COMPLETE $($Config.Tag)"
}

