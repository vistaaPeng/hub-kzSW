# 必须在所有导入之前设置环境变量
import os
# 使用 Hugging Face 镜像站（国内更稳定）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 设置更长的超时时间（5分钟）
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
# 禁用 Telemetry
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

"""
人民日报 NER 训练脚本 - 支持 BERT+Linear 和 BERT+CRF 两种模型架构

================================================================================
                          模型架构说明
================================================================================
两种模型结构对比：

  ┌─────────────────────────────────────────────────────────────────┐
  │                    BERT+Linear（基线模型）                      │
  ├─────────────────────────────────────────────────────────────────┤
  │  输入文本 → BERT Encoder →  token embedding [B,T,H]            │
  │                                      │                         │
  │                                      ▼                         │
  │                                Linear Layer [B,T,C]            │
  │                                      │                         │
  │                                      ▼                         │
  │                           argmax → 标签序列                    │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │                      BERT+CRF（进阶模型）                       │
  ├─────────────────────────────────────────────────────────────────┤
  │  输入文本 → BERT Encoder →  token embedding [B,T,H]            │
  │                                      │                         │
  │                                      ▼                         │
  │                                Linear Layer [B,T,C]            │
  │                                      │                         │
  │                                      ▼                         │
  │                           CRF Layer（转移矩阵学习）             │
  │                                      │                         │
  │                                      ▼                         │
  │                           Viterbi → 最优标签序列                │
  └─────────────────────────────────────────────────────────────────┘

CRF 的优势：
  1. 学习标签转移约束（如 B-PER 后面更可能是 I-PER 而不是 O）
  2. 全局优化：考虑整个序列的标签组合，避免局部最优
  3. 自动捕获 BIO 标注规范（如 I-tag 不能单独出现）

================================================================================
                          核心技术点详解
================================================================================
1. 分层学习率（Layer-wise Learning Rate）
   - BERT 层：2e-5（预训练权重已经包含丰富的语言知识，只需微调）
   - 分类头：1e-4（随机初始化，需要更大学习率快速收敛）
   - 原理：预训练模型在大规模语料上学习到了通用语言表示，过大的学习率会破坏这些知识

2. Linear Warmup 学习率调度
   - 训练初期学习率从 0 线性增长到设定值
   - 作用：防止训练初期大梯度破坏 BERT 预训练权重
   - 默认 warmup_ratio=0.1，即前 10% 的训练步数用于 warmup

3. 梯度累积（Gradient Accumulation）
   - 每 grad_accum 个 batch 累积梯度，然后一起更新参数
   - 等效于 batch_size × grad_accum 的效果
   - 适用于显存有限的场景，用小 batch_size 模拟大 batch

4. Entity-level F1 评估
   - 使用 seqeval 库计算，而非简单的 token-level accuracy
   - 要求实体的边界和类型完全正确才算命中
   - 例如：预测 "B-PER I-PER" 而真实是 "B-PER" 不算正确
   - 是 NER 任务的标准评测指标

5. 梯度裁剪（Gradient Clipping）
   - 将梯度范数限制在 max_norm=1.0 以内
   - 防止梯度爆炸，稳定训练过程

6. L2 正则化（Weight Decay）
   - weight_decay=0.01 对所有权重施加 L2 正则
   - 防止模型过拟合

================================================================================
                          使用方式
================================================================================
python train_peoples_daily.py                        # 训练 BERT+Linear（基线）
python train_peoples_daily.py --use_crf              # 训练 BERT+CRF（带转移约束）
python train_peoples_daily.py --epochs 5 --lr 3e-5   # 自定义超参数

依赖安装：
  pip install torch transformers seqeval pytorch-crf tqdm
"""

# ============================================================
# 0. 环境变量设置
# ============================================================
# macOS/Linux 上 OpenMP 库可能与 PyTorch 冲突，设置此变量避免报错
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ============================================================
# 1. 标准库 / 第三方库导入
# ============================================================
import json           # 用于保存训练日志（JSON 格式）
import time           # 用于统计每个 epoch 的训练耗时
import argparse       # 命令行参数解析
from pathlib import Path  # 路径管理，比 os.path 更优雅

import torch
import torch.nn as nn
from torch.optim import AdamW  # AdamW 优化器（带权重衰减的 Adam）
from transformers import BertTokenizer, get_linear_schedule_with_warmup  # BERT 分词器 + 线性 warmup 调度器
from tqdm import tqdm  # 进度条

# 从本地模块导入：标签体系构建 & DataLoader 构建
from dataset_peoples_daily import build_label_schema, build_dataloaders
# 从本地模块导入：模型工厂函数
from model import build_model

# ============================================================
# 2. 全局路径常量
# ============================================================
# 项目根目录：src/ 的上一级，即 序列标注项目/
ROOT = Path(__file__).parent.parent
# 预训练 BERT 模型配置：
#   - 如果本地有模型，设置为本地路径（如 "pretrain_models/bert-base-chinese"）
#   - 如果本地没有模型，设置为 Hugging Face Hub 模型名（如 "bert-base-chinese"），会自动下载
# 注意：Windows 本地路径需要特殊处理，推荐使用相对路径或直接使用 Hub 模型名
BERT_PATH = "bert-base-chinese"  # 直接使用 Hugging Face Hub 模型，自动下载
# 人民日报数据集目录
DATA_DIR = ROOT / "data" / "peoples_daily"
# 模型检查点保存目录
CKPT_DIR = ROOT / "outputs" / "checkpoints"
# 训练日志保存目录
LOG_DIR = ROOT / "outputs" / "logs"


# ============================================================
# 3. 评估函数：每个 epoch 结束后在验证集上计算 loss 和 entity-level F1
# ============================================================
def evaluate_epoch(
    model: nn.Module,
    loader,                  # 验证集 DataLoader
    id2label: dict,          # 标签 ID → 标签名 的映射字典
    device: torch.device,
    use_crf: bool,           # 是否使用 CRF 模型
) -> tuple[float, float]:
    """
    在验证集 loader 上评估模型，返回 (平均 loss, entity-level F1)。

    entity-level F1 的含义：
      - 只有当一个实体的 **边界（B/I标签）和类型** 完全正确时才算正确
      - 比 token-level accuracy 严格得多，是 NER 任务的标准评测指标
    """
    # seqeval 是专门用于序列标注评测的库，计算 entity-level 的 P/R/F1
    from seqeval.metrics import f1_score as seqeval_f1

    model.eval()  # 切换到评估模式：关闭 dropout，冻结 BatchNorm 等
    total_loss = 0.0
    all_preds: list[list[str]] = []   # 存储所有样本的预测标签序列
    all_golds: list[list[str]] = []   # 存储所有样本的真实标签序列

    with torch.no_grad():  # 评估时不需要计算梯度，节省显存和计算
        for batch in loader:
            # ---------- 数据搬到 GPU/CPU ----------
            input_ids      = batch["input_ids"].to(device)       # token ID 序列
            attention_mask  = batch["attention_mask"].to(device)  # 1=真实token, 0=padding
            token_type_ids  = batch["token_type_ids"].to(device)  # 句子A/B标记（单句全0）
            labels          = batch["labels"].to(device)          # 标签ID序列，padding位=-100

            # ---------- 前向传播（两种模型分支）----------
            if use_crf:
                # CRF 模型：返回 (emission 分数, CRF 负对数似然 loss)
                emissions, loss = model(input_ids, attention_mask, token_type_ids, labels)
                # decode 使用 Viterbi 算法解码最优标签路径
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                # Linear 模型：返回 (logits [B,T,num_labels], 交叉熵 loss)
                logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
                # 取概率最大的标签作为预测
                pred_ids_list = logits.argmax(dim=-1).tolist()

            total_loss += loss.item()

            # ---------- 将 ID 转回标签字符串 ----------
            labels_np = labels.cpu().tolist()
            for i in range(len(input_ids)):  # 遍历 batch 中每个样本
                gold_seq = []  # 该样本的真实标签序列
                pred_seq = []  # 该样本的预测标签序列
                token_labels = labels_np[i]
                pred_ids = pred_ids_list[i]

                for j, gold_id in enumerate(token_labels):
                    # -100 是 padding 位置，不参与评估
                    if gold_id == -100:
                        continue
                    # 真实标签：ID → 字符串（如 "B-PER"）
                    gold_seq.append(id2label[gold_id])
                    # 预测标签：ID → 字符串，越界则默认 "O"
                    if use_crf:
                        if j < len(pred_ids):
                            pred_seq.append(id2label.get(pred_ids[j], "O"))
                        else:
                            pred_seq.append("O")
                    else:
                        pred_seq.append(id2label.get(pred_ids[j], "O"))

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    # 计算平均 loss 和 entity-level F1
    avg_loss = total_loss / len(loader)
    entity_f1 = seqeval_f1(all_golds, all_preds)
    return avg_loss, entity_f1


# ============================================================
# 4. 训练函数：训练一个 epoch
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader,              # 训练集 DataLoader
    optimizer,           # 优化器（AdamW）
    scheduler,           # 学习率调度器（Linear Warmup）
    device: torch.device,
    epoch: int,          # 当前 epoch 编号（从 1 开始）
    total_epochs: int,   # 总 epoch 数
    grad_accum: int,     # 梯度累积步数
) -> float:
    """
    训练一个 epoch，返回平均 loss。

    梯度累积（Gradient Accumulation）：
      - 当 batch_size 受限于显存时，可以用 grad_accum > 1 来模拟更大的 batch
      - 例如 grad_accum=4, batch_size=8 → 等效 batch_size=32
      - 每累积 grad_accum 步才执行一次 optimizer.step()
    """
    model.train()  # 切换到训练模式：启用 dropout 等
    total_loss = 0.0
    optimizer.zero_grad()  # 开始前清零梯度

    # tqdm 进度条，显示当前 epoch 和实时 loss
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        # ---------- 数据搬到设备 ----------
        input_ids      = batch["input_ids"].to(device)
        attention_mask  = batch["attention_mask"].to(device)
        token_type_ids  = batch["token_type_ids"].to(device)
        labels          = batch["labels"].to(device)

        # ---------- 前向传播 ----------
        # model 返回值第一个是 logits/emissions（训练时不用），第二个是 loss
        _, loss = model(input_ids, attention_mask, token_type_ids, labels)

        # ---------- 反向传播（带梯度累积缩放）----------
        # 除以 grad_accum 保证等效梯度大小与大 batch 一致
        (loss / grad_accum).backward()
        total_loss += loss.item()

        # ---------- 每 grad_accum 步更新一次参数 ----------
        if (step + 1) % grad_accum == 0:
            # 梯度裁剪：防止梯度爆炸（将梯度范数限制在 1.0 以内）
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()     # 更新参数
            scheduler.step()     # 更新学习率
            optimizer.zero_grad()  # 清零梯度，为下一轮累积做准备

        # 在进度条上显示当前 batch 的 loss
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 处理最后不足 grad_accum 的残留批次（例如总共有 99 个 batch，grad_accum=4，最后 3 个不会触发更新）
    remainder = len(loader) % grad_accum
    if remainder != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


# ============================================================
# 5. 主函数：组装整个训练流程
# ============================================================
def main():
    # ---------- 5.1 解析命令行参数 ----------
    args = parse_args()

    # ---------- 5.2 设备选择：优先使用 GPU ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    # ---------- 5.3 构建 BIO 标签体系 ----------
    # BIO 标注规范说明：
    #   - B-tag：实体的开始（Begin），如 B-PER 表示人名开始
    #   - I-tag：实体的内部（Inside），如 I-PER 表示人名中间/结尾
    #   - O：非实体（Outside）
    # 实体类型（人民日报数据集）：
    #   PER（人名）、ORG（组织机构）、LOC（地点）
    # 返回值：
    #   labels:    标签名列表，如 ["O", "B-PER", "I-PER", "B-ORG", ...]
    #   label2id:  标签名 → ID，如 {"O": 0, "B-PER": 1, ...}
    #   id2label:  ID → 标签名，如 {0: "O", 1: "B-PER", ...}
    labels, label2id, id2label = build_label_schema()
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}（O + {len(labels) - 1} 个实体标签）")

    # ---------- 5.4 加载 BERT 分词器 ----------
    # BertTokenizer 会将中文文本拆分为字级别 token，并映射为 ID
    # 使用 Hugging Face Hub 模型名，自动下载并缓存
    # 添加参数避免不必要的网络请求
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_path,
        trust_remote_code=False,
        use_fast=True,
        local_files_only=False
    )

    # ---------- 5.5 构建 DataLoader ----------
    # build_dataloaders 内部流程：
    #   1. 读取 JSON 格式的数据集文件
    #   2. Tokenize：将文本转换为 token ID 序列
    #   3. 标签对齐：处理子词切分后的标签映射问题
    #   4. Padding/Truncation：统一序列长度到 max_length
    #   5. 构建 PyTorch DataLoader，返回训练集/验证集/测试集
    train_loader, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length,   # 最大序列长度（超过截断，不足 padding）
        data_dir=DATA_DIR,
    )

    # ---------- 5.6 构建模型 ----------
    # build_model 是一个工厂函数，根据 use_crf 参数返回不同模型：
    #   - use_crf=True:   BERT + Dropout + Linear + CRF
    #   - use_crf=False:  BERT + Dropout + Linear
    # 模型结构：
    #   BERT 输出 [B,T,H] → Dropout → Linear [B,T,C] → (CRF) → 标签序列
    #   H: BERT 隐藏层维度（768）
    #   C: 标签类别数（BIO 标签总数）
    model = build_model(
        use_crf=args.use_crf,
        bert_path=args.bert_path,  # 使用 Hub 模型名
        num_labels=num_labels,
        dropout=args.dropout,
    ).to(device)  # 将模型移动到 GPU/CPU

    # ---------- 5.7 分层学习率设置 ----------
    # 核心思想：BERT 预训练层已经学到了很好的语言表示，用小学习率微调即可；
    #           分类头（Linear/CRF）是随机初始化的，需要较大学习率快速收敛。
    bert_params = list(model.bert.parameters())  # BERT 编码器的所有参数
    head_params = (
        list(model.classifier.parameters()) +    # 线性分类头参数
        list(model.dropout.parameters()) +        # dropout 层参数
        (list(model.crf.parameters()) if args.use_crf else [])  # CRF 层参数（如有）
    )
    optimizer = AdamW(
        [
            # BERT 层：使用基础学习率 args.lr（默认 2e-5）
            {"params": bert_params, "lr": args.lr},
            # 分类头：使用 args.lr * args.head_lr_mult（默认 2e-5 * 5 = 1e-4）
            {"params": head_params, "lr": args.lr * args.head_lr_mult},
        ],
        weight_decay=0.01,  # L2 正则化，防止过拟合
    )

    # ---------- 5.8 学习率调度器：Linear Warmup ----------
    # total_steps：整个训练过程的总更新步数
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    # warmup_steps：前 warmup_ratio 比例的步数用于 warmup
    warmup_steps = int(total_steps * args.warmup_ratio)
    # 线性 warmup + 线性衰减调度器：
    #   - 前 warmup_steps 步：学习率从 0 线性增长到设定值
    #   - 之后的学习率：线性衰减到 0
    #   - 目的：训练初期 BERT 参数对梯度敏感，小学习率避免破坏预训练权重
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"\n训练步数：{total_steps}，预热步数：{warmup_steps}")

    # ---------- 5.9 准备保存路径和日志 ----------
    run_tag = "crf" if args.use_crf else "linear"  # 用于区分两种模型的文件名后缀
    CKPT_DIR.mkdir(parents=True, exist_ok=True)     # 确保目录存在
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"best_{run_tag}_peoples_daily.pt"    # 最优模型保存路径
    log_path = LOG_DIR / f"train_{run_tag}_peoples_daily.json"    # 训练日志保存路径

    best_f1 = 0.0       # 记录最优 entity-level F1
    log_records = []     # 训练日志记录

    # ---------- 5.10 训练主循环 ----------
    print(f"\n开始训练（{'BERT+CRF' if args.use_crf else 'BERT+Linear'}）...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # 训练一个 epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs, args.grad_accum
        )
        # 在验证集上评估
        val_loss, val_f1 = evaluate_epoch(model, val_loader, id2label, device, args.use_crf)
        elapsed = time.time() - t0

        # 打印本 epoch 的训练结果
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_entity_f1={val_f1:.4f} | "
            f"time={elapsed:.0f}s"
        )

        # 记录到日志列表
        log_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_entity_f1": round(val_f1, 6),
            "elapsed_s": round(elapsed, 1),
        })

        # ---------- 保存最优模型 ----------
        # 模型选择策略：选择验证集上 entity-level F1 最高的模型
        # 为什么不用 loss 选择？因为 loss 是 token-level 的，而 F1 是实体级别的，更贴近任务目标
        if val_f1 > best_f1:
            best_f1 = val_f1
            # 保存检查点，包含完整的模型信息：
            torch.save(
                {
                    "epoch": epoch,                  # 训练到第几个 epoch
                    "use_crf": args.use_crf,         # 是否使用 CRF（评估时需要匹配）
                    "state_dict": model.state_dict(), # 模型权重（最重要）
                    "val_entity_f1": val_f1,          # 该检查点对应的验证集 F1
                    "label2id": label2id,             # 标签名→ID 映射（推理时需要）
                    "id2label": id2label,             # ID→标签名映射（推理时需要）
                    "args": vars(args),               # 训练参数（方便复现和推理）
                },
                ckpt_path,
            )
            print(f"  ★ 新最优 F1={val_f1:.4f}，已保存 → {ckpt_path}")

    # ---------- 5.11 训练结束，保存日志 ----------
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    # 打印最终结果和后续步骤提示
    print(f"\n训练完成！最优 val_entity_f1={best_f1:.4f}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  训练日志:   {log_path}")
    print(f"\n下一步：python evaluate_peoples_daily.py {'--use_crf' if args.use_crf else ''}")


# ============================================================
# 6. 命令行参数解析
# ============================================================
def parse_args():
    """定义并解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练人民日报 NER 模型")
    # --use_crf：布尔开关，指定则使用 CRF 层，否则使用 Linear 分类头
    parser.add_argument("--use_crf", action="store_true", help="使用 CRF 层（否则使用线性头）")
    # --bert_path：预训练 BERT 模型的本地路径
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    # --epochs：训练轮数
    parser.add_argument("--epochs", type=int, default=3)
    # --batch_size：每个 batch 的样本数
    parser.add_argument("--batch_size", type=int, default=32)
    # --max_length：输入序列最大长度（token 数），超过截断，不足 padding
    parser.add_argument("--max_length", type=int, default=128)
    # --lr：BERT 层的基础学习率
    parser.add_argument("--lr", type=float, default=2e-5, help="BERT 层学习率")
    # --head_lr_mult：分类头学习率相对于 BERT 层的倍数
    # 例如 mult=5, lr=2e-5 → 分类头 lr=1e-4
    parser.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头学习率倍数")
    # --warmup_ratio：warmup 步数占总步数的比例
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    # --grad_accum：梯度累积步数，用于在小 batch_size 下模拟大 batch
    parser.add_argument("--grad_accum", type=int, default=1)
    # --dropout：分类头的 dropout 比率
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


# ============================================================
# 7. 程序入口
# ============================================================
if __name__ == "__main__":
    main()
