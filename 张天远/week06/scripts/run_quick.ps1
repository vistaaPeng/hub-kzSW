# run_quick.ps1 — 快速验证（1 epoch / 1000条，总耗时 ~20min）
# 用于改动代码后快速确认没有 bug

$startTime = Get-Date
Write-Host "快速验证模式 — 1 epoch, 1000条 SFT" -ForegroundColor Cyan

python src/train.py --bert_path bert-base-chinese --epochs 1 --batch_size 16 --max_length 64
python src/train.py --bert_path bert-base-chinese --epochs 1 --batch_size 16 --max_length 64 --use_class_weight
python src_llm/train_sft.py --model_path Qwen/Qwen2-0.5B-Instruct --num_train 1000 --epochs 1
python src_llm/classify_llm.py --model_path Qwen/Qwen2-0.5B-Instruct --demo
python src_llm/evaluate_sft.py --model_path Qwen/Qwen2-0.5B-Instruct --demo
python src/evaluate.py --pool cls --bert_path bert-base-chinese
python src/compare_class_weight.py --pool cls --bert_path bert-base-chinese

Write-Host "快速验证完成 — $([math]::Round(((Get-Date) - $startTime).TotalMinutes, 1)) min" -ForegroundColor Green
