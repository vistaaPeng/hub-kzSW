# src_llm: 大模型 NER 序列标注

本目录使用 OpenAI 兼容的 Chat Completions API 做 one-shot/few-shot NER 推理，并支持用本地开源因果语言模型做 LoRA/QLoRA 监督微调。数据默认读取上一层 `data/train.json`、`data/validation.json`、`data/test.json` 和 `label_names.json`。

## 安装依赖

```powershell
cd C:\Users\sean\Projects\hub-kzSW\尹文武\第七周作业\src_llm
pip install -r requirements.txt
```

QLoRA 依赖 `bitsandbytes` 的 4bit 量化能力。Windows 原生环境如果安装失败，建议改用 WSL/Linux，或先使用 `--mode lora`。

## API 配置

默认使用 OpenAI 兼容接口调用 `deepseek-v4-flash`：

```powershell
$env:OPENAI_API_KEY="你的key"
$env:OPENAI_BASE_URL="https://api.deepseek.com"
$env:OPENAI_MODEL="deepseek-v4-flash"
```

也可以在命令行中传入：

```powershell
python main.py --mode few_shot --api_key "你的key" --base_url "https://api.deepseek.com" --api_model deepseek-v4-flash
```

## One-shot 验证

```powershell
python main.py --mode one_shot --eval_split validation --eval_limit 20
```

`one_shot` 会从训练集抽 1 条样本作为示例，然后调用大模型输出每个 token 的 BIO 标签。

## Few-shot 验证

```powershell
python main.py --mode few_shot --num_shots 5 --eval_split validation --eval_limit 50
```

常用参数：

- `--num_shots`: prompt 中放入的训练样本数量。
- `--eval_split`: `train`、`validation` 或 `test`。
- `--eval_limit`: 快速验证时限制样本数。
- `--temperature`: 生成温度，NER 建议保持 `0`。
- `--max_new_tokens`: 单条预测最多生成 token 数。

## LoRA 训练

```powershell
python main.py ^
  --mode lora ^
  --model_name_or_path C:\path\to\chat-model ^
  --train_limit 1000 ^
  --epochs 1 ^
  --batch_size 1 ^
  --gradient_accumulation_steps 8 ^
  --lr 2e-4 ^
  --output_dir ..\outputs\src_llm\lora_ner
```

训练完成后脚本会默认加载 adapter 在验证集上评估。只训练不验证可以加：

```powershell
python main.py --mode lora --model_name_or_path C:\path\to\chat-model --skip_eval_after_train
```

## QLoRA 训练

```powershell
python main.py ^
  --mode qlora ^
  --model_name_or_path C:\path\to\chat-model ^
  --train_limit 1000 ^
  --epochs 1 ^
  --batch_size 1 ^
  --gradient_accumulation_steps 8 ^
  --lr 2e-4 ^
  --output_dir ..\outputs\src_llm\qlora_ner
```

QLoRA 与 LoRA 参数一致，但会以 4bit 方式加载基础模型，显存占用更低。

## 加载已训练 Adapter 验证

```powershell
python main.py ^
  --mode validate_adapter ^
  --model_name_or_path C:\path\to\chat-model ^
  --adapter_path ..\outputs\src_llm\lora_ner ^
  --eval_split test ^
  --eval_limit 100
```

## 输出

预测文件会保存到 `--output_dir`：

```text
{mode}_{eval_split}_predictions.json
```

文件中包含实体级 precision、recall、F1，以及每条样本的 `tokens`、`gold`、`pred` 和模型原始输出。

## 参数速查

```text
--mode one_shot|few_shot|lora|qlora|validate_adapter
--data_dir ../data
--eval_split validation
--eval_limit 50
--train_limit 1000
--num_shots 5
--api_model deepseek-v4-flash
--model_name_or_path <本地或 HuggingFace 模型路径>
--adapter_path <LoRA/QLoRA adapter 路径>
--lora_r 8
--lora_alpha 16
--lora_dropout 0.05
--target_modules q_proj,k_proj,v_proj,o_proj
```
