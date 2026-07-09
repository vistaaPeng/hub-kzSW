# 文本匹配实验结果汇总（work8）

## 一、任务说明

在 **LCQMC** 与 **BQ Corpus** 两个数据集上，使用 BERT 预训练模型完成中文语义文本匹配，对比三种方案：

- BiEncoder + CosineEmbeddingLoss
- BiEncoder + TripletLoss
- CrossEncoder + CrossEntropyLoss

- **预训练模型**：bert-base-chinese（4 层）
- **训练配置**：1 epoch，batch_size=16，lr=2e-5
- **评估集**：test（两个数据集均有标签）
- **硬件环境**：NVIDIA GeForce GTX 1060 6GB

> 课堂演示在 `src/` 上用 AFQMC 跑了 3 epoch；本实验因 LCQMC 数据量大（23.8 万条），统一采用 **1 epoch** 以控制训练时间。

---

## 二、数据集规模

| 数据集 | train | validation | test |
|--------|------:|-----------:|-----:|
| LCQMC | 238,766 | 8,802 | 12,500 |
| BQ Corpus | 68,960 | 8,620 | 8,620 |

---

## 三、BQ Corpus — test 集结果

| 方法 | Accuracy | F1(weighted) | F1(正例) | 备注 |
|------|--------:|-------------:|--------:|------|
| BiEncoder + Cosine | 0.8073 | 0.8070 | 0.8163 | threshold=0.67，AUC=0.880 |
| BiEncoder + Triplet | 0.8194 | 0.8193 | 0.8235 | threshold=0.55，AUC=0.898 |
| **CrossEncoder** | **0.8354** | **0.8354** | **0.8350** | argmax |

### 训练过程（1 epoch）

| 方法 | 训练损失 | val F1 | 耗时 |
|------|--------:|-------:|-----:|
| BiEncoder + Cosine | 0.220 | 0.816 | 977s（~16 min） |
| BiEncoder + Triplet | 0.107 | 0.820 | 695s（~12 min） |
| CrossEncoder | 0.472（acc=0.770） | 0.836 | 858s（~14 min） |

---

## 四、LCQMC — test 集结果

| 方法 | Accuracy | F1(weighted) | F1(正例) | 备注 |
|------|--------:|-------------:|--------:|------|
| BiEncoder + Cosine | 0.8221 | 0.8221 | 0.8231 | threshold=0.97，AUC=0.905 |
| BiEncoder + Triplet | 0.8400 | 0.8400 | 0.8376 | threshold=0.83，AUC=0.917 |
| **CrossEncoder** | **0.8489** | **0.8475** | **0.8619** | argmax |

### 训练过程（1 epoch）

| 方法 | 训练损失 | val F1 | 耗时 |
|------|--------:|-------:|-----:|
| BiEncoder + Cosine | 0.189 | 0.761 | 3283s（~55 min） |
| BiEncoder + Triplet | 0.026 | 0.795 | 2752s（~46 min） |
| CrossEncoder | 0.272（acc=0.884） | 0.833 | 2921s（~49 min） |

> LCQMC 单方法训练时间约为 BQ 的 3~4 倍，与训练集规模（23.8 万 vs 6.9 万）基本成正比。

---

## 五、跨数据集对比分析

### 5.1 方法对比

**CrossEncoder 在两个数据集上均为最优。**

| 数据集 | Cosine | Triplet | CrossEncoder | 最佳 |
|--------|-------:|--------:|-------------:|------|
| BQ test F1(w) | 0.807 | 0.819 | **0.835** | CrossEncoder |
| LCQMC test F1(w) | 0.822 | 0.840 | **0.848** | CrossEncoder |

- **CrossEncoder vs BiEncoder**：交互型模型让两句在 BERT 每一层充分交互，表达能力更强；两个数据集上 F1 均领先 BiEncoder 约 **1.5~2.7 个百分点**。
- **Triplet vs Cosine**：与 AFQMC 演示（Cosine 略优）相反，**BQ 和 LCQMC 上 Triplet 均优于 Cosine**（BQ +1.2pt，LCQMC +1.8pt）。可能原因：这两个数据集训练量更大（尤其 LCQMC 正样本对约 12 万），三元组数量充足，TripletLoss 的相对距离约束更有效。
- **BiEncoder 阈值差异**：LCQMC Cosine 最优阈值高达 **0.97**，说明在大数据训练后正负样本相似度整体偏高，需更高切点才能区分；BQ 阈值相对正常（0.55~0.67）。

### 5.2 数据集对比

| 方法 | BQ test F1(w) | LCQMC test F1(w) | 差距 |
|------|-------------:|-----------------:|-----:|
| BiEncoder + Cosine | 0.807 | 0.822 | +1.5pt |
| BiEncoder + Triplet | 0.819 | 0.840 | +2.1pt |
| CrossEncoder | 0.835 | 0.848 | +1.3pt |

- **LCQMC 整体优于 BQ**：同一方法在 LCQMC 上 F1  consistently 高约 1~2 点。
- **可能原因**：LCQMC 训练集更大（23.8 万 vs 6.9 万），模型见到更多句对；BQ 为金融客服领域，问句表述更口语化、边界 case 更多，匹配难度略高。
- **训练效率**：BQ 三种方法合计约 **40 分钟**；LCQMC 合计约 **2.5 小时**，适合先用 BQ 验证流程再跑 LCQMC。

### 5.3 与 AFQMC（src/）对照

| 数据集/方法 | F1(正例) 或 F1(w) | 说明 |
|------------|------------------|------|
| AFQMC / BiEncoder Cosine（src/ 参考） | F1(w)=0.6765 | validation 集，3 epoch |
| BQ / BiEncoder Cosine（本实验） | F1(w)=0.807 | test 集，1 epoch |
| LCQMC / BiEncoder Cosine（本实验） | F1(w)=0.822 | test 集，1 epoch |

本实验 F1 高于 AFQMC 参考值，不能直接横向比绝对数值（数据集、评估划分、epoch 数均不同）。可确认的规律与 `src/` 一致：**CrossEncoder 精度最高**；差异在于 AFQMC 上 Cosine ≥ Triplet，而 BQ/LCQMC 上 **Triplet > Cosine**，说明 TripletLoss 对数据规模更敏感。

---

## 六、结论

1. **最佳方案**：两个数据集上 **CrossEncoder** 均为最优（BQ F1=0.835，LCQMC F1=0.848）；若需向量检索场景，可选 BiEncoder + Triplet 作为精度与效率的折中。
2. **Loss 选择**：BQ/LCQMC 大数据下 **TripletLoss 优于 CosineEmbeddingLoss**；AFQMC 小正样本场景则 Cosine 更稳——数据量是关键变量。
3. **数据集差异**：LCQMC 因训练集更大，同配置下 F1 普遍高于 BQ 约 1~2 点；金融域 BQ 匹配难度略高。
4. **工程取舍**：BiEncoder 可预计算向量、适合 RAG 召回；CrossEncoder 精度更高但每对需完整过 BERT，适合精排。
5. **可改进方向**：增加 epoch（LCQMC 仅 1 epoch，val→test 仍有 gap）；尝试 12 层 BERT；BiEncoder 做难负样本挖掘以缩小与 CrossEncoder 的差距。

---

*原始数据：`outputs/comparison_test.json`，训练日志：`outputs/{lcqmc,bq_corpus}/logs/train_*.json`*
