# 文本匹配训练与评估项目

## 项目结构
- `data_analysis.py`：统计数据集基本信息并输出图表。
- `train.py`：训练文本匹配模型，可切换 BiEncoder / CrossEncoder。
- `evaluate.py`：加载保存的模型并输出评估指标。
- `model.py`：模型定义与损失函数。
- `text_matching_utils.py`：数据加载与通用工具函数。
- `outputs/figures/`：数据分析图表。
- `outputs/logs/`：训练日志与曲线图。
- `outputs/checkpoints/`：最佳模型保存目录。
- `outputs/reports/`：评估报告。

## 数据格式说明
当前项目支持以下数据格式：
- JSONL，每行一个 JSON 对象，字段包括：
  - `sentence1` / `sentence2`（或 `text1` / `text2`）
  - `label`（0/1）
- CSV / TSV，列名可包含：
  - `sentence1` / `sentence2` / `label`
  - `text1` / `text2` / `label`

## 数据分析命令
```bash
python data_analysis.py --data_path data/bq_corpus/train.jsonl --output_dir outputs
```

## 训练命令示例
```bash
python train.py \
  --data_path data/bq_corpus/train.jsonl \
  --model_type biencoder \
  --loss_type cosine \
  --epochs 2 \
  --batch_size 16
```

```bash
python train.py \
  --data_path data/bq_corpus/train.jsonl \
  --model_type crossencoder \
  --loss_type cross_entropy \
  --epochs 2 \
  --batch_size 8
```

## 评估命令示例
```bash
python evaluate.py \
  --data_path data/bq_corpus/test.jsonl \
  --model_path outputs/checkpoints/best_model.pt
```
