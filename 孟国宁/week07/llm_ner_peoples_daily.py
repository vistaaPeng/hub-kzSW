"""
使用大模型 API 做人民日报 NER：zero-shot vs few-shot 对比

教学重点：
  1. LLM 做 NER 的 prompt 设计
     - zero-shot：只靠任务描述，无样例
     - few-shot：给 3 个标注示例，引导格式对齐
  2. 结构化输出解析（JSON提取 + 容错处理）
  3. LLM 的 span 级别 F1 计算（与 BERT 保持可比性）
  4. 成本控制：只采样 100 条，不跑完整验证集

使用方式：
  python llm_ner_peoples_daily.py
  python llm_ner_peoples_daily.py --n_samples 50 --model qwen-max

依赖：
  pip install openai
  export DASHSCOPE_API_KEY="sk-xxx"
"""

# ============================================================
# 第一部分：导入依赖库
# ============================================================

import os
# 设置环境变量，防止 OpenMP 库冲突导致的多线程错误
# 这是 macOS/Linux 上常见的兼容性问题解决方案
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json        # JSON 解析和序列化
import time        # 时间相关操作，用于 API 重试时的延时
import random      # 随机数生成，用于数据采样
import argparse    # 命令行参数解析
import re          # 正则表达式，用于从 LLM 输出中提取 JSON
from pathlib import Path  # 路径操作，更优雅的文件路径处理
from collections import defaultdict  # 带默认值的字典，用于按类型分组

# 导入 OpenAI 官方 SDK（兼容通义千问等兼容 OpenAI 接口的服务）
from openai import OpenAI
# 导入 dotenv 库，从 .env 文件加载环境变量
from dotenv import load_dotenv
# 加载 .env 文件中的环境变量（如 DASHSCOPE_API_KEY）
load_dotenv()

# ============================================================
# 第二部分：路径和常量定义
# ============================================================

# 项目根目录：当前文件的父目录的父目录（即项目根目录）
ROOT = Path(__file__).parent.parent
# 数据目录：存放人民日报 NER 数据集
DATA_DIR = ROOT / "data" / "peoples_daily"
# 日志目录：存放评估结果
LOG_DIR = ROOT / "outputs" / "logs"

# 人民日报 NER 数据集的实体类型映射
# 键：英文缩写（用于模型输出和计算）
# 值：中文名称（用于展示和理解）
ENTITY_TYPE_ZH = {
    "PER": "人名",      # Person - 人名实体
    "ORG": "组织机构",   # Organization - 组织机构实体
    "LOC": "地点",      # Location - 地点实体
}

# 实体类型英文列表，用于遍历和验证
ENTITY_TYPES_EN = list(ENTITY_TYPE_ZH.keys())


# ============================================================
# 第三部分：API 客户端初始化
# ============================================================

def build_client() -> OpenAI:
    """
    构建 OpenAI 兼容的 API 客户端。

    使用通义千问（DashScope）的兼容接口，这样可以用 OpenAI SDK
    调用国产大模型，无需学习新的 API 格式。

    Returns:
        OpenAI: 配置好的 API 客户端实例

    Raises:
        EnvironmentError: 如果未设置 DASHSCOPE_API_KEY 环境变量
    """
    # 从环境变量获取 API 密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")

    # 创建 OpenAI 客户端，指定通义千问的兼容接口地址
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


# ============================================================
# 第四部分：Gold 标准数据处理
# ============================================================

def gold_spans_from_record(record: dict) -> set[tuple[str, str, int, int]]:
    """
    从标注数据中提取 gold 标准的实体 span。

    将 BIO 标签序列转换为实体集合，便于后续与模型预测结果对比。

    BIO 标签说明：
      - B-XXX：实体开始（Beginning）
      - I-XXX：实体内部（Inside）
      - O：非实体（Outside）

    Args:
        record: 包含 'tokens' 和 'ner_tags' 的字典
                - tokens: 分词后的字符串列表
                - ner_tags: 对应的 BIO 标签列表

    Returns:
        set of tuple: 实体集合，每个元素为 (实体文本, 实体类型, 起始位置, 结束位置)
                      位置是 token 级别的索引

    示例：
        输入 tokens=["北", "京", "大", "学"], ner_tags=["B-ORG", "I-ORG", "I-ORG", "I-ORG"]
        输出 {("北京大学", "ORG", 0, 3)}
    """
    spans = set()  # 存储提取的实体 span
    tokens = record["tokens"]      # 分词结果
    ner_tags = record["ner_tags"]  # BIO 标签序列

    # 当前正在构建的实体信息
    current_entity = []      # 当前实体的 token 列表
    current_type = None      # 当前实体的类型（如 PER, ORG, LOC）
    current_start = None     # 当前实体的起始位置索引

    # 遍历每个 token 及其对应的标签
    for i, (token, tag) in enumerate(zip(tokens, ner_tags)):

        if tag.startswith("B-"):
            # 情况1：遇到 B- 标签，表示新实体开始
            # 先保存之前积累的实体（如果有的话）
            if current_entity and current_type:
                entity_text = "".join(current_entity)  # 拼接 token 得到实体文本
                spans.add((entity_text, current_type, current_start, i - 1))

            # 开始记录新实体
            current_entity = [token]      # 初始化实体 token 列表
            current_type = tag[2:]        # 提取实体类型（去掉 "B-" 前缀）
            current_start = i             # 记录起始位置

        elif tag.startswith("I-") and current_type == tag[2:]:
            # 情况2：遇到 I- 标签，且类型与当前实体一致
            # 表示当前实体的延续
            current_entity.append(token)

        else:
            # 情况3：遇到 O 标签，或 I- 标签但类型不匹配
            # 表示当前实体结束
            if current_entity and current_type:
                entity_text = "".join(current_entity)
                spans.add((entity_text, current_type, current_start, i - 1))

            # 重置状态，准备处理下一个实体
            current_entity = []
            current_type = None
            current_start = None

    # 处理序列末尾的最后一个实体（如果有的话）
    if current_entity and current_type:
        entity_text = "".join(current_entity)
        spans.add((entity_text, current_type, current_start, len(tokens) - 1))

    return spans


# ============================================================
# 第五部分：LLM 预测结果解析
# ============================================================

def pred_spans_from_response(text: str, response_text: str) -> set[tuple[str, str, int, int]]:
    """
    从 LLM 的 JSON 输出中解析预测的实体 span。

    处理流程：
      1. 用正则表达式提取 JSON 块（兼容 markdown 代码块格式）
      2. 解析 JSON，提取 entities 数组
      3. 在原文中定位每个实体的位置

    Args:
        text: 原始输入文本，用于定位实体在原文中的位置
        response_text: LLM 的输出文本，期望包含 JSON 格式的实体列表

    Returns:
        set of tuple: 预测的实体集合，每个元素为 (实体文本, 实体类型, 起始位置, 结束位置)

    容错处理：
      - 如果 JSON 不存在或格式错误，返回空集合
      - 如果实体类型不在预定义范围内，跳过该实体
      - 如果实体文本在原文中找不到，跳过该实体
    """
    # 步骤1：用正则表达式提取 JSON 块
    # r"\{.*\}" 匹配最外层的花括号及其内容
    # re.DOTALL 让 . 也能匹配换行符
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        return set()  # 没有找到 JSON，返回空集合

    # 步骤2：解析 JSON
    try:
        obj = json.loads(json_match.group())
    except json.JSONDecodeError:
        return set()  # JSON 格式错误，返回空集合

    # 步骤3：提取 entities 数组
    entities = obj.get("entities", [])
    if not isinstance(entities, list):
        return set()  # entities 不是列表，返回空集合

    # 步骤4：遍历实体，定位在原文中的位置
    spans = set()
    for ent in entities:
        # 跳过非字典类型的元素
        if not isinstance(ent, dict):
            continue

        # 提取实体文本和类型
        surface = str(ent.get("text", "")).strip()  # 实体文本（去空格）
        etype = str(ent.get("type", "")).strip()     # 实体类型

        # 验证：实体文本非空，且类型在预定义范围内
        if not surface or etype not in ENTITY_TYPES_EN:
            continue

        # 在原文中查找实体位置（取第一次出现的位置）
        idx = text.find(surface)
        if idx == -1:
            continue  # 实体在原文中找不到，跳过

        # 计算结束位置并添加到结果集
        spans.add((surface, etype, idx, idx + len(surface) - 1))

    return spans


# ============================================================
# 第六部分：评估指标计算
# ============================================================

def compute_span_f1(all_golds: list[set], all_preds: list[set]) -> dict:
    """
    计算 span 级别的精确率（Precision）、召回率（Recall）和 F1 值。

    使用集合交集运算计算真正例（TP），与 BERT 模型的评估方式保持一致。

    指标说明：
      - Precision = TP / 预测总数 （预测为实体的有多少是正确的）
      - Recall = TP / 标注总数 （标注的实体有多少被预测出来）
      - F1 = 2 * P * R / (P + R) （精确率和召回率的调和平均）

    Args:
        all_golds: 每条样本的 gold 标准实体集合列表
        all_preds: 每条样本的预测实体集合列表

    Returns:
        dict: 包含 precision, recall, f1, tp, pred_total, gold_total 的字典
    """
    # 计算真正例（TP）：预测与标注完全匹配的实体数量
    # 使用集合交集 g & p 获取匹配的实体
    tp = sum(len(g & p) for g, p in zip(all_golds, all_preds))

    # 预测的实体总数
    pred_total = sum(len(p) for p in all_preds)
    # 标注的实体总数
    gold_total = sum(len(g) for g in all_golds)

    # 计算精确率：预测正确的比例
    p = tp / pred_total if pred_total else 0.0
    # 计算召回率：标注实体被找到的比例
    r = tp / gold_total if gold_total else 0.0
    # 计算 F1 值：精确率和召回率的调和平均
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    return {
        "precision": p,       # 精确率
        "recall": r,          # 召回率
        "f1": f1,             # F1 值
        "tp": tp,             # 真正例数量
        "pred_total": pred_total,  # 预测总数
        "gold_total": gold_total   # 标注总数
    }


# ============================================================
# 第七部分：Prompt 模板设计
# ============================================================

# 系统提示词（System Prompt）
# 定义 LLM 的角色和任务，以及输出格式要求
SYSTEM_PROMPT = """你是一个命名实体识别（NER）专家，专门处理中文文本。
请从用户输入的文本中识别以下3类实体，并以 JSON 格式输出结果：
- PER：人名
- ORG：组织机构名称
- LOC：地点名称

输出格式（严格遵守，不要包含其他文字）：
{"entities": [{"text": "实体文本", "type": "实体类型英文名"}, ...]}

如果没有实体，输出：{"entities": []}"""

# Few-shot 示例列表
# 提供 3 个标注示例，帮助模型理解任务格式和实体边界
FEW_SHOT_EXAMPLES = [
    {
        "text": "海钓比赛地点在厦门与金门之间的海域。",
        "output": '{"entities": [{"text": "厦门", "type": "LOC"}, {"text": "金门", "type": "LOC"}]}'
    },
    {
        "text": "中国国家主席习近平在北京人民大会堂会见了美国总统奥巴马。",
        "output": '{"entities": [{"text": "习近平", "type": "PER"}, {"text": "北京", "type": "LOC"}, {"text": "人民大会堂", "type": "LOC"}, {"text": "奥巴马", "type": "PER"}]}'
    },
    {
        "text": "北京大学和清华大学是中国最著名的两所大学。",
        "output": '{"entities": [{"text": "北京大学", "type": "ORG"}, {"text": "清华大学", "type": "ORG"}]}'
    },
]


def zero_shot_prompt(text: str) -> list[dict]:
    """
    构建 zero-shot 提示词。

    Zero-shot 模式：不提供任何示例，仅依靠系统提示词描述任务。
    测试模型在没有示例指导下的理解能力。

    Args:
        text: 需要进行 NER 的输入文本

    Returns:
        list[dict]: OpenAI Chat API 格式的消息列表
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},  # 系统提示词
        {"role": "user", "content": text},              # 用户输入的待处理文本
    ]


def few_shot_prompt(text: str) -> list[dict]:
    """
    构建 few-shot 提示词。

    Few-shot 模式：提供 3 个标注示例，帮助模型理解：
      1. 输出格式（JSON 结构）
      2. 实体类型（PER, ORG, LOC）
      3. 实体边界（如何识别实体的起止位置）

    Args:
        text: 需要进行 NER 的输入文本

    Returns:
        list[dict]: OpenAI Chat API 格式的消息列表，包含示例对话
    """
    # 初始化消息列表，以系统提示词开始
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 添加 few-shot 示例（每个示例是一组 user-assistant 对话）
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["text"]})        # 示例输入
        messages.append({"role": "assistant", "content": ex["output"]}) # 期望输出

    # 最后添加用户实际要处理的文本
    messages.append({"role": "user", "content": text})

    return messages


# ============================================================
# 第八部分：API 调用
# ============================================================

def call_api(client: OpenAI, messages: list[dict], model: str) -> str:
    """
    调用 LLM API，获取模型输出。

    使用指数退避策略进行重试，提高调用成功率。

    Args:
        client: OpenAI API 客户端
        messages: Chat 格式的消息列表
        model: 模型名称（如 "qwen-plus"）

    Returns:
        str: 模型输出的文本内容，失败时返回空字符串
    """
    # 最多重试 3 次
    for attempt in range(3):
        try:
            # 调用 Chat Completion API
            resp = client.chat.completions.create(
                model=model,          # 模型名称
                messages=messages,    # 消息列表
                temperature=0.0,      # 温度设为 0，使用确定性输出
                max_tokens=512,       # 最大输出 token 数
            )
            # 返回第一个响应的文本内容
            return resp.choices[0].message.content or ""

        except Exception as e:
            # 如果不是最后一次尝试，等待后重试
            if attempt < 2:
                # 指数退避：第 1 次等 1 秒，第 2 次等 2 秒
                time.sleep(2 ** attempt)
            else:
                # 最后一次尝试也失败，打印错误信息
                print(f"  API 调用失败：{e}")
                return ""

    return ""


# ============================================================
# 第九部分：数据采样
# ============================================================

def sample_records(n: int, seed: int = 42) -> list[dict]:
    """
    从验证集中采样 n 条记录，尽量覆盖所有实体类型。

    使用分层采样策略：
      1. 先按实体类型分组
      2. 每种类型均匀采样
      3. 如果不够 n 条，从剩余数据中随机补充

    Args:
        n: 需要采样的记录数量
        seed: 随机种子，保证结果可复现

    Returns:
        list[dict]: 采样后的记录列表
    """
    # 加载验证集数据
    with open(DATA_DIR / "validation.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    # 设置随机种子，保证结果可复现
    random.seed(seed)

    # 步骤1：按实体类型分组
    # by_type 字典：key=实体类型, value=包含该类型实体的记录列表
    by_type = defaultdict(list)
    for r in records:
        for tag in r.get("ner_tags", []):
            if tag.startswith("B-"):
                etype = tag[2:]  # 提取实体类型
                by_type[etype].append(r)

    # 步骤2：分层采样
    selected = set()        # 已选记录的 id 集合（用于去重）
    selected_list = []      # 已选记录的有序列表

    # 计算每种类型应该采样的数量
    per_type = max(1, n // len(ENTITY_TYPES_EN))

    # 对每种实体类型进行采样
    for etype in ENTITY_TYPES_EN:
        # 获取包含当前类型实体的候选记录（排除已选的）
        candidates = [r for r in by_type[etype] if id(r) not in selected]
        # 随机采样
        chosen = random.sample(candidates, min(per_type, len(candidates)))
        # 添加到已选列表
        for r in chosen:
            if len(selected_list) < n and id(r) not in selected:
                selected.add(id(r))
                selected_list.append(r)

    # 步骤3：如果不够 n 条，从剩余数据中随机补充
    remaining = [r for r in records if id(r) not in selected]
    random.shuffle(remaining)
    for r in remaining:
        if len(selected_list) >= n:
            break
        selected_list.append(r)

    return selected_list[:n]


# ============================================================
# 第十部分：数据转换
# ============================================================

def record_to_text(record: dict) -> str:
    """
    将记录转换为纯文本。

    将分词结果拼接成完整的文本字符串，用于输入到 LLM。

    Args:
        record: 包含 'tokens' 的记录字典

    Returns:
        str: 拼接后的完整文本
    """
    return "".join(record["tokens"])


# ============================================================
# 第十一部分：命令行参数解析
# ============================================================

def parse_args():
    """
    解析命令行参数。

    支持的参数：
      --n_samples: 采样数量，默认 100
      --model: 模型名称，默认 "qwen-plus"

    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="LLM zero-shot/few-shot 人民日报 NER 对比"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="采样验证集的数量（默认 100）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-plus",
        help="使用的模型名称（默认 qwen-plus）"
    )
    return parser.parse_args()


# ============================================================
# 第十二部分：主函数
# ============================================================

def main():
    """
    主函数：执行 LLM NER 评估流程。

    流程：
      1. 解析命令行参数
      2. 初始化 API 客户端
      3. 采样验证集数据
      4. 对每条数据分别进行 zero-shot 和 few-shot 预测
      5. 计算两种方法的评估指标
      6. 输出对比结果并保存日志
    """
    # 步骤1：解析命令行参数
    args = parse_args()

    # 步骤2：初始化 API 客户端
    client = build_client()

    # 步骤3：采样验证集数据
    records = sample_records(args.n_samples)
    print(f"采样 {len(records)} 条验证集样本")

    # 步骤4：初始化存储列表
    zero_shot_golds = []   # zero-shot 的 gold 标准
    zero_shot_preds = []   # zero-shot 的预测结果
    few_shot_golds = []    # few-shot 的 gold 标准
    few_shot_preds = []    # few-shot 的预测结果
    detail_records = []    # 详细记录（用于分析）

    # 步骤5：遍历每条记录进行预测
    for i, record in enumerate(records, 1):
        # 将记录转换为文本
        text = record_to_text(record)
        # 提取 gold 标准实体
        gold = gold_spans_from_record(record)

        # --- Zero-shot 预测 ---
        # 构建 zero-shot 提示词
        zs_resp = call_api(client, zero_shot_prompt(text), args.model)
        # 从 LLM 输出中解析预测结果
        zs_pred = pred_spans_from_response(text, zs_resp)

        # --- Few-shot 预测 ---
        # 构建 few-shot 提示词
        fs_resp = call_api(client, few_shot_prompt(text), args.model)
        # 从 LLM 输出中解析预测结果
        fs_pred = pred_spans_from_response(text, fs_resp)

        # 保存结果
        zero_shot_golds.append(gold)
        zero_shot_preds.append(zs_pred)
        few_shot_golds.append(gold)
        few_shot_preds.append(fs_pred)

        # 保存详细记录（便于后续分析）
        detail_records.append({
            "text": text,
            "gold": [{"text": s, "type": t} for s, t, _, _ in gold],
            "zero_shot": [{"text": s, "type": t} for s, t, _, _ in zs_pred],
            "few_shot": [{"text": s, "type": t} for s, t, _, _ in fs_pred],
        })

        # 每处理 10 条或处理完最后一条时，打印进度
        if i % 10 == 0 or i == len(records):
            print(f"  已处理 {i}/{len(records)} 条")

    # 步骤6：计算评估指标
    zs_metrics = compute_span_f1(zero_shot_golds, zero_shot_preds)
    fs_metrics = compute_span_f1(few_shot_golds, few_shot_preds)

    # 步骤7：输出对比结果
    print("\n" + "=" * 60)
    print(f"LLM NER 对比结果（模型：{args.model}，样本：{len(records)} 条）")
    print("=" * 60)
    print(f"{'方案':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    print(f"{'Zero-shot':<20} {zs_metrics['precision']:>10.4f} {zs_metrics['recall']:>10.4f} {zs_metrics['f1']:>10.4f}")
    print(f"{'Few-shot (3例)':<20} {fs_metrics['precision']:>10.4f} {fs_metrics['recall']:>10.4f} {fs_metrics['f1']:>10.4f}")

    # 步骤8：保存评估结果到日志文件
    LOG_DIR.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    result = {
        "model": args.model,
        "n_samples": len(records),
        "zero_shot": zs_metrics,
        "few_shot": fs_metrics,
        "detail": detail_records,
    }

    # 确保数值可 JSON 序列化（处理 numpy 类型）
    def _to_python(v):
        """将 numpy 类型转换为 Python 原生类型"""
        return v.item() if hasattr(v, "item") else v

    result["zero_shot"] = {k: _to_python(v) for k, v in result["zero_shot"].items()}
    result["few_shot"] = {k: _to_python(v) for k, v in result["few_shot"].items()}

    # 写入 JSON 文件
    out_path = LOG_DIR / "eval_llm_peoples_daily.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nLLM 评估结果已保存 → {out_path}")
    print("\n下一步：python compare_results.py")


# ============================================================
# 第十三部分：程序入口
# ============================================================

if __name__ == "__main__":
    # 当脚本直接运行时（非被导入时），执行主函数
    main()
