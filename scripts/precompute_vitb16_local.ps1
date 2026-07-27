$ErrorActionPreference = "Stop"
$Python = "D:\Anaconda_envs\YOLO\python.exe"
$Script = "D:\456\project\scripts\precompute_clip_features.py"
$OutputRoot = "D:\456\data\processed\clip_features_vitb16"
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
    $Output = "$OutputRoot\$($Job[0])_vitb16_openai.npz"
    if (Test-Path -LiteralPath $Output) { continue }
    & $Python $Script --domain-root $Job[1] --output $Output --model ViT-B-16 --weights openai `
        --batch-size 8 --num-workers 2 --write-source-labels
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

