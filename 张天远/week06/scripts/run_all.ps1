$ErrorActionPreference = "Continue"
Write-Host "=== 1/6 BERT CrossEntropy ==="
python src/train.py --bert_path bert-base-chinese --epochs 3 --batch_size 16 --max_length 64
Write-Host "=== 2/6 BERT Weighted ==="
python src/train.py --bert_path bert-base-chinese --epochs 3 --batch_size 16 --max_length 64 --use_class_weight
Write-Host "=== 3/6 LoRA ==="
python src_llm/train_sft.py --model_path Qwen/Qwen2-0.5B-Instruct --num_train 5000 --epochs 3
Write-Host "=== 4/6 Zero-shot ==="
python src_llm/classify_llm.py --model_path Qwen/Qwen2-0.5B-Instruct --num_samples 200 --seed 42
Write-Host "=== 5/6 SFT Eval ==="
python src_llm/evaluate_sft.py --model_path Qwen/Qwen2-0.5B-Instruct --num_samples 200 --seed 42
Write-Host "=== 6/6 Eval ==="
python src/evaluate.py --pool cls --bert_path bert-base-chinese
python src/evaluate.py --pool cls_weighted --bert_path bert-base-chinese
python src/compare_class_weight.py --pool cls --bert_path bert-base-chinese
Write-Host "=== Done ==="
