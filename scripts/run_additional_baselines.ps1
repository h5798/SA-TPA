param(
    [string]$RunName = "additional_baselines_v1"
)

$ErrorActionPreference = "Stop"
$Python = "D:\Anaconda_envs\YOLO\python.exe"
$Project = "D:\456\project"
$FeatureRoot = "D:\456\data\processed\clip_features"
$Methods = @("t3a", "tip_adapter_source")
$Benchmarks = @(
    [pscustomobject]@{
        Name="office31"; ResultRoot="D:\456\results\office31";
        Tasks=@(
            @("A2W","amazon","webcam"), @("D2W","dslr","webcam"), @("W2A","webcam","amazon"),
            @("A2D","amazon","dslr"), @("D2A","dslr","amazon"), @("W2D","webcam","dslr")
        )
    },
    [pscustomobject]@{
        Name="officehome"; ResultRoot="D:\456\results\officehome";
        Tasks=@(
            @("A2C","art","clipart"), @("A2P","art","product"), @("A2R","art","real_world"),
            @("C2A","clipart","art"), @("C2P","clipart","product"), @("C2R","clipart","real_world"),
            @("P2A","product","art"), @("P2C","product","clipart"), @("P2R","product","real_world"),
            @("R2A","real_world","art"), @("R2C","real_world","clipart"), @("R2P","real_world","product")
        )
    }
)

foreach ($Benchmark in $Benchmarks) {
    $ResultsCsv = "$($Benchmark.ResultRoot)\$RunName.csv"
    foreach ($Task in $Benchmark.Tasks) {
        foreach ($Method in $Methods) {
            $TaskName = $Task[0]
            $Source = "$FeatureRoot\$($Benchmark.Name)_$($Task[1])_vitb32_openai.npz"
            $Target = "$FeatureRoot\$($Benchmark.Name)_$($Task[2])_vitb32_openai.npz"
            $Prediction = "$($Benchmark.ResultRoot)\predictions\$RunName\$TaskName`_$Method.npz"
            if (Test-Path -LiteralPath $Prediction) { continue }
            & $Python "$Project\scripts\run_task.py" --run-tag $RunName --task $TaskName --method $Method `
                --source-features $Source --target-features $Target --prediction-output $Prediction `
                --results-csv $ResultsCsv | Out-Null
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        Write-Output "COMPLETE $($Benchmark.Name) $($Task[0])"
    }
}

