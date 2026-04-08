# Adaptive Day/Night Overnight Training Script
# ================================================
# Runs 3 training configurations sequentially on the RTX 4070.
# Each run tests a different strategy to fix catastrophic forgetting
# (the model forgetting daytime detection after nighttime fine-tuning).
#
# Estimated total time: ~8-9 hours (run before sleeping!)
#
# HOW TO USE:
#   1. Open PowerShell as Administrator
#   2. cd E:\persona-detection-main  (or wherever your project is)
#   3. .\training_scripts\train_overnight.ps1
#
# AFTER TRAINING:
#   Compare the 3 best.pt files visually on your afternoon CCTV clip.
#   The winner is the one that scores highest AND detects 30+ persons in daytime.

$ErrorActionPreference = "Stop"
$StartTime = Get-Date
$MergedDir = "D:\persona_detection\datasets\merged"
$Project   = "F:\persona-detection-main\runs\detect"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host " OVERNIGHT TRAINING — 3 SEQUENTIAL RUNS" -ForegroundColor Cyan
Write-Host " Started: $StartTime" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# ─────────────────────────────────────────────
# RUN A: Conservative fine-tune
#   freeze=15 (deep freeze), lr0=0.0005 (gentle)
#   Goal: Maximum preservation of daytime person detection
# ─────────────────────────────────────────────
Write-Host "`n[RUN A] freeze=15 | lr0=0.0005 | Gentle fine-tune" -ForegroundColor Yellow
Write-Host "   Goal: Preserve daytime detection while improving night"
python training_scripts/train_yolo.py `
    --merged-dir  $MergedDir `
    --project     $Project `
    --name-prefix "yolov8s_runA_" `
    --base-model  "yolov8s.pt" `
    --freeze      15 `
    --lr0         0.0005 `
    --epochs      50 `
    --batch       16 `
    --imgsz       640 `
    --workers     4 `
    --device      cuda `
    --only-rotation 0

if ($LASTEXITCODE -ne 0) {
    Write-Host "[RUN A] FAILED with exit code $LASTEXITCODE" -ForegroundColor Red
} else {
    Write-Host "[RUN A] COMPLETE" -ForegroundColor Green
}

# ─────────────────────────────────────────────
# RUN B: Head-only training (maximum freeze)
#   freeze=20 (only the detection head trains)
#   Goal: Backbone fully preserved, only output layers adapt to night
# ─────────────────────────────────────────────
Write-Host "`n[RUN B] freeze=20 | lr0=0.001 | Head-only fine-tune" -ForegroundColor Yellow
Write-Host "   Goal: Backbone untouched — strongest daytime preservation"
python training_scripts/train_yolo.py `
    --merged-dir  $MergedDir `
    --project     $Project `
    --name-prefix "yolov8s_runB_" `
    --base-model  "yolov8s.pt" `
    --freeze      20 `
    --lr0         0.001 `
    --epochs      50 `
    --batch       16 `
    --imgsz       640 `
    --workers     4 `
    --device      cuda `
    --only-rotation 0

if ($LASTEXITCODE -ne 0) {
    Write-Host "[RUN B] FAILED with exit code $LASTEXITCODE" -ForegroundColor Red
} else {
    Write-Host "[RUN B] COMPLETE" -ForegroundColor Green
}

# ─────────────────────────────────────────────
# RUN C: Very slow learning rate
#   freeze=10 (same as original), but lr0=0.0003 (very slow)
#   Goal: Original depth but so gentle it barely changes daytime weights
# ─────────────────────────────────────────────
Write-Host "`n[RUN C] freeze=10 | lr0=0.0003 | Very slow LR" -ForegroundColor Yellow
Write-Host "   Goal: Original depth but minimal catastrophic forgetting"
python training_scripts/train_yolo.py `
    --merged-dir  $MergedDir `
    --project     $Project `
    --name-prefix "yolov8s_runC_" `
    --base-model  "yolov8s.pt" `
    --freeze      10 `
    --lr0         0.0003 `
    --epochs      60 `
    --batch       16 `
    --imgsz       640 `
    --workers     4 `
    --device      cuda `
    --only-rotation 0

if ($LASTEXITCODE -ne 0) {
    Write-Host "[RUN C] FAILED with exit code $LASTEXITCODE" -ForegroundColor Red
} else {
    Write-Host "[RUN C] COMPLETE" -ForegroundColor Green
}

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
$EndTime  = Get-Date
$Duration = $EndTime - $StartTime
Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host " ALL RUNS COMPLETE" -ForegroundColor Cyan
Write-Host " Total time: $($Duration.Hours)h $($Duration.Minutes)m" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Test each best.pt visually on afternoon + night CCTV clips"
Write-Host "  2. Run yolo_param_sweep.py with each --data nightowls_val.yaml"
Write-Host "  3. Pick the run with highest mAP50 AND best daytime recall"
Write-Host ""
Write-Host "Best weights locations:"
Write-Host "  Run A: $Project\yolov8s_runA_0\weights\best.pt"
Write-Host "  Run B: $Project\yolov8s_runB_0\weights\best.pt"
Write-Host "  Run C: $Project\yolov8s_runC_0\weights\best.pt"
