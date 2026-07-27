param(
    [string]$RunName = "confirmatory_v1"
)

$ErrorActionPreference = "Stop"
$Python = "D:\Anaconda_envs\YOLO\python.exe"
$Project = "D:\456\project"
$FeatureRoot = "D:\456\data\processed\clip_features"
$ResultRoot = "D:\456\results\officehome"
$ResultsCsv = "$ResultRoot\$RunName.csv"
$Lock = Get-Content -LiteralPath "$Project\protocols\locked_hyperparameters.json" | ConvertFrom-Json
if (-not $Lock.locked_for_officehome) { throw "Office-Home parameters are not locked" }

$Methods = @(
    "clip_zero_shot", "prompt_ensemble", "source_prototype", "source_anchored_text",
    "no_source_anchor", "satpa_no_uncertainty", "satpa"
)
$Tasks = @(
    @("A2C", "art", "clipart"), @("A2P", "art", "product"), @("A2R", "art", "real_world"),
    @("C2A", "clipart", "art"), @("C2P", "clipart", "product"), @("C2R", "clipart", "real_world"),
    @("P2A", "product", "art"), @("P2C", "product", "clipart"), @("P2R", "product", "real_world"),
    @("R2A", "real_world", "art"), @("R2C", "real_world", "clipart"), @("R2P", "real_world", "product")
)

foreach ($Task in $Tasks) {
    foreach ($Method in $Methods) {
        $TaskName = $Task[0]
        $Source = "$FeatureRoot\officehome_$($Task[1])_vitb32_openai.npz"
        $Target = "$FeatureRoot\officehome_$($Task[2])_vitb32_openai.npz"
        $Prediction = "$ResultRoot\predictions\$RunName\$TaskName`_$Method.npz"
        if (Test-Path -LiteralPath $Prediction) { continue }
        & $Python "$Project\scripts\run_task.py" --run-tag $RunName --task $TaskName --method $Method `
            --source-features $Source --target-features $Target --prediction-output $Prediction `
            --results-csv $ResultsCsv --alpha-source $Lock.alpha_source --alpha-target $Lock.alpha_target `
            --confidence-threshold $Lock.confidence_threshold --top-k $Lock.top_k `
            --class-prior-strength $Lock.class_prior_strength | Out-Null
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    Write-Output "COMPLETE $($Task[0])"
}

