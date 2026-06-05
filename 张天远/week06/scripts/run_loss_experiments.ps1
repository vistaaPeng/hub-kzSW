$ErrorActionPreference = "Continue"

Write-Host "1/3 Soft Weighted (sqrt)"
python src/train.py --bert_path bert-base-chinese --pool cls --epochs 3 --loss_type soft

Write-Host "2/3 Focal Loss (gamma=2)"
python src/train.py --bert_path bert-base-chinese --pool cls --epochs 3 --loss_type focal

Write-Host "3/3 Two-stage (plain + freeze)"
python src/train.py --bert_path bert-base-chinese --pool cls --epochs 3 --loss_type plain --freeze_bert

Write-Host "=== Evaluate Soft ==="
python src/evaluate.py --pool cls --loss_type soft --bert_path bert-base-chinese
if (Test-Path outputs/figures/confusion_cls.png) {
    Copy-Item outputs/figures/confusion_cls.png outputs/figures/confusion_soft.png
}

Write-Host "=== Evaluate Focal ==="
python src/evaluate.py --pool cls --loss_type focal --bert_path bert-base-chinese
if (Test-Path outputs/figures/confusion_cls.png) {
    Copy-Item outputs/figures/confusion_cls.png outputs/figures/confusion_focal.png
}

Write-Host "=== Evaluate Freeze ==="
python src/evaluate.py --pool cls --loss_type plain --ckpt_path outputs/checkpoints/best_cls_plain_freeze.pt --bert_path bert-base-chinese
if (Test-Path outputs/figures/confusion_cls.png) {
    Copy-Item outputs/figures/confusion_cls.png outputs/figures/confusion_freeze.png
}

Write-Host "=== Done ==="
Write-Host "保存的文件:"
Write-Host "  train_log_soft.json / confusion_soft.png"
Write-Host "  train_log_focal.json / confusion_focal.png"
Write-Host "  train_log_plain_freeze.json / confusion_freeze.png"
