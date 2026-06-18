import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Optional

# ----------------------------- 辅助函数 -----------------------------
def extract_entity_types_from_samples(samples: List[Dict]) -> List[str]:
    """从样本列表中提取所有唯一的实体类型（去除B-/I-前缀）"""
    types = set()
    for sample in samples:
        for tag in sample['ner_tags']:
            if tag != 'O' and tag.startswith(('B-', 'I-')):
                ent_type = tag[2:]
                types.add(ent_type)
    return sorted(types)

def extract_entities_from_bio(tokens: List[str], tags: List[str]) -> List[Dict]:
    """从BIO标签序列提取实体列表"""
    entities = []
    i = 0
    n = len(tokens)
    while i < n:
        tag = tags[i]
        if tag.startswith('B-'):
            ent_type = tag[2:]
            start = i
            j = i + 1
            while j < n and tags[j] == f'I-{ent_type}':
                j += 1
            end = j - 1
            ent_text = ''.join(tokens[start:end+1])
            entities.append({"text": ent_text, "type": ent_type})
            i = j
        else:
            i += 1
    return entities

def entities_to_json(entities: List[Dict]) -> str:
    """实体列表转JSON字符串"""
    return json.dumps({"entities": entities}, ensure_ascii=False)

# ----------------------------- SFTDataset 类 -----------------------------
class SFTDataset(Dataset):
    """
    用于LLM微调的NER数据集（指令微调格式），自动构建 system prompt，
    使用 chat_template 编码对话，构造 labels（仅 assistant 部分参与损失计算）。
    """
    def __init__(self,
                 data: List[Dict],
                 tokenizer: AutoTokenizer,
                 max_length: int = 512,
                 entity_types: Optional[List[str]] = None):
        """
        Args:
            data: 样本列表，每个样本包含 "tokens" 和 "ner_tags"
            tokenizer: HuggingFace tokenizer（需支持 chat_template）
            max_length: 最大序列长度
            entity_types: 可选，实体类型列表。若不提供则从 data 中自动提取。
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 确定实体类型
        if entity_types is None:
            self.entity_types = extract_entity_types_from_samples(data)
        else:
            self.entity_types = entity_types

        # 构建 system prompt
        self.system_prompt = self._build_system_prompt()

        # 预处理所有样本
        self.examples = []
        for sample in data:
            tokens = sample['tokens']
            tags = sample['ner_tags']
            user_text = ''.join(tokens)
            entities = extract_entities_from_bio(tokens, tags)
            assistant_text = entities_to_json(entities)
            encoded = self._encode_chat(user_text, assistant_text)
            self.examples.append(encoded)

    def _build_system_prompt(self) -> str:
        type_list = ", ".join(self.entity_types)
        return (
            "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
            f"实体类型（英文标识）：{type_list}\n"
            '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
            '无实体时输出：{"entities": []}'
        )

    def _encode_chat(self, user_text: str, assistant_text: str) -> Dict[str, torch.Tensor]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text}
        ]

        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]

        # 定位 assistant 部分
        assistant_encoding = self.tokenizer(
            assistant_text,
            add_special_tokens=False,
            return_tensors="pt"
        )
        assistant_ids = assistant_encoding['input_ids'][0].tolist()
        input_ids_list = input_ids.tolist()
        win_len = len(assistant_ids)
        assistant_start = None
        for i in range(len(input_ids_list) - win_len + 1):
            if input_ids_list[i:i+win_len] == assistant_ids:
                assistant_start = i
                break

        labels = torch.full_like(input_ids, -100)
        if assistant_start is not None:
            labels[assistant_start:assistant_start+win_len] = input_ids[assistant_start:assistant_start+win_len]
        else:
            # 降级（理论上不会发生）
            labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# ----------------------------- collate_fn -----------------------------
def collate_fn(batch: List[Dict], pad_token_id: int):
    """
    动态 padding，将 batch 内所有样本填充到相同长度。
    Args:
        batch: list of dict，每个 dict 包含 'input_ids', 'attention_mask', 'labels' (1D tensor)
        pad_token_id: 用于填充 input_ids 的 token id
    Returns:
        dict with keys 'input_ids', 'attention_mask', 'labels'，每个值为 (batch_size, max_len) 的 tensor
    """
    max_len = max([item['input_ids'].size(0) for item in batch])
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for item in batch:
        seq_len = item['input_ids'].size(0)
        pad_len = max_len - seq_len

        # input_ids padding
        input_ids_pad = torch.cat([item['input_ids'], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        # attention_mask padding
        attn_pad = torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)])
        # labels padding
        labels_pad = torch.cat([item['labels'], torch.full((pad_len,), -100, dtype=torch.long)])

        padded_input_ids.append(input_ids_pad)
        padded_attention_mask.append(attn_pad)
        padded_labels.append(labels_pad)

    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_mask),
        'labels': torch.stack(padded_labels)
    }

# ----------------------------- 使用示例 -----------------------------
if __name__ == '__main__':
    # 假设外部已经读取数据
    with open('train.json', 'r', encoding='utf-8') as f:
        train_raw = json.load(f)
    with open('validation.json', 'r', encoding='utf-8') as f:
        val_raw = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 训练集：自动提取实体类型
    train_dataset = SFTDataset(train_raw, tokenizer, max_length=512)
    # 验证集：使用训练集的实体类型（确保标签空间一致）
    val_dataset = SFTDataset(val_raw[:300], tokenizer, max_length=512, entity_types=train_dataset.entity_types)

    from torch.utils.data import DataLoader
    from functools import partial

    # collate_fn 绑定 pad_token_id
    collate_fn = partial(ner_collate_fn, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for batch in train_loader:
        print(batch['input_ids'].shape)  # [4, max_len_in_batch]
        break