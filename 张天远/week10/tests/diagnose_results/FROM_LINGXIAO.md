From: 凌霄 (LXC 101)
To: 灵枢 (Windows)
Date: 2026-07-06

---

## 凌宵的诊断报告 — rag_scratch 项目

### Bug（紧急）

文件 src/glossary.py 的 search_glossary 函数在第201行用了 Path() 但函数内没有 import。
Path 只在 build_glossary_index 函数内 import（第127行），作用域不共享。

修复：在文件顶部的全局 import 区域加一行 from pathlib import Path

影响：这个 bug 导致 app.py 在 POST /query 时调 search_glossary 报 NameError，
被 try/except 静默吞掉后降级回静态50条注入。
动态 Glossary 从上线起就没真正生效过。

### 诊断发现

#### 1. MRR=1.00 是测试集设计假象
40道手动题中，30题(75%)的关键词直接出现在 top-1 chunk 的 heading 里。
剩10道难题（m08/m12/m13/m25/m26/m27/m31/m32/m34/m38）才是真正考验语义匹配的。

建议：评估报告把简单题和难题分开列。

#### 2. 延迟瓶颈在 LLM 生成
P50 延迟分解：
  - LLM 生成：4.41s（69%）
  - 查询重写：1.04s（16%）
  - Glossary 检索：0.37s
  - 向量/BM25 检索：0.02s 各
  - 父块扩展：0.001s
  - 总计：~6s

检索管线全部加起来不到0.5s。要优化只能打 LLM。

#### 3. 自评估 bias
Faithfulness 评估器和 generator 都是 DeepSeek v4-flash。
如果用另一个模型（比如 LXC106 的 Qwen2.5-7B）做 evaluator，得到的分数更可信。

#### 4. 测试集缺负例
MRR=1.00 不可能是检索完美——所有问题都有完美匹配的文档。
没有设计无答案的负例（答非所问检测这一环节没测过）。

### 诊断脚本位置
E:\npl\workspaces\npl_tran\rag_scratch\tests\diagnose_rag.py
  - 可单独跑 diagnose_mrr_bias()（0 API）
  - 可单独跑 diagnose_latency()（5 API）
  - 还有 diagnose_evaluator_gap() 和 diagnose_failure_modes() 需要 API

数据文件在 tests\diagnose_results\ 下。
