# BQ 两阶段检索 — PowerShell 版
# 用法: .\scripts\run_bq_twostage.ps1
$ErrorActionPreference = "Stop"
Set-Location E:\npl\workspaces\npl_tran\text_match

Write-Host "=== Step 1: 存档 AFQMC checkpoint ===" -ForegroundColor Cyan
Copy-Item outputs\checkpoints\biencoder_cosine_best.pt outputs\checkpoints\_bk_afqmc_bi.pt -Force
Copy-Item outputs\checkpoints\crossencoder_best.pt       outputs\checkpoints\_bk_afqmc_ce.pt -Force
Write-Host "  OK"

Write-Host "=== Step 2: 换入 BQ checkpoint ===" -ForegroundColor Cyan
Copy-Item outputs\checkpoints\biencoder_cosine_bq_corpus_best.pt outputs\checkpoints\biencoder_cosine_best.pt -Force
Copy-Item outputs\checkpoints\crossencoder_bq_corpus_best.pt   outputs\checkpoints\crossencoder_best.pt       -Force
Write-Host "  OK"

Write-Host "=== Step 3: 运行 BQ 两阶段检索 ===" -ForegroundColor Cyan
conda run -n py312 python src/two_stage_retrieval.py --k 100 --data_dir data/bq_corpus

Write-Host ""
Write-Host "=== Step 4: 恢复 AFQMC checkpoint ===" -ForegroundColor Cyan
Copy-Item outputs\checkpoints\_bk_afqmc_bi.pt outputs\checkpoints\biencoder_cosine_best.pt -Force
Copy-Item outputs\checkpoints\_bk_afqmc_ce.pt outputs\checkpoints\crossencoder_best.pt       -Force
Remove-Item outputs\checkpoints\_bk_afqmc_bi.pt
Remove-Item outputs\checkpoints\_bk_afqmc_ce.pt
Write-Host "  OK"

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "结果: outputs\logs\two_stage_bq_corpus.json"
