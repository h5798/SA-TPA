param(
    [switch]$RunFeatures
)

$ErrorActionPreference = "Stop"
$ProjectRoot = "D:\456\project"
$Python = "D:\Anaconda_envs\YOLO\python.exe"

& $Python "$ProjectRoot\scripts\environment_check.py"
& $Python "$ProjectRoot\scripts\validate_data.py"

if ($RunFeatures) {
    $Jobs = @(
        @("office31_amazon", "D:\456\data\raw\office31\Office-31\amazon"),
        @("office31_dslr", "D:\456\data\raw\office31\Office-31\dslr"),
        @("office31_webcam", "D:\456\data\raw\office31\Office-31\webcam"),
        @("officehome_art", "D:\456\data\raw\officehome\OfficeHomeDataset\Art"),
        @("officehome_clipart", "D:\456\data\raw\officehome\OfficeHomeDataset\Clipart"),
        @("officehome_product", "D:\456\data\raw\officehome\OfficeHomeDataset\Product"),
        @("officehome_real_world", "D:\456\data\raw\officehome\OfficeHomeDataset\Real World")
    )
    foreach ($Job in $Jobs) {
        & $Python "$ProjectRoot\scripts\precompute_clip_features.py" `
            --domain-root $Job[1] `
            --output "D:\456\data\processed\clip_features\$($Job[0])_vitb32_openai.npz"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
