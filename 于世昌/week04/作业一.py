"""
尝试用pytorch实现一个transformer层。
"""
import torch
import numpy as np
from torch import nn
import math

from safetensors.torch import load_file  # 读取safetensors权重
from transformers import BertModel, BertConfig

    
def bert_transformer_diy_weight(input, weight_dict):
    """输入一个128×768的矩阵， 按照transformer原理进行处理
       一、multi-head self-attention
       1、基于列分割12个 128 × 64 的矩阵
       2、基于 bert的model拿到 q k v 3个linear
       Q: w1x + b1
       K: w2x + b2
       V: w3x + b3
       把这6个参数拿到，然后 基于输入input  128 × 768 进行矩阵运算 拿到3个 128×768的矩阵
       3、把 Q K V 3个矩阵分别按列切分12份
       4、Q的每一份128×64， 和 K的每一份 128×64的转置进行矩阵运算， 得到12个128×128矩阵
       5、然后这12个128×128的矩阵，按行进行 每一项/更号d_k ， d_k = emb_dim / n_head
       我们这里 emb_dim = 768，  n_head = 12， 768/12 = 64
       6、然后128×128矩阵进行 行softmax
       7、然后这12个 128 × 128的矩阵， 挨个 和 V的 128×64 做 矩阵运算， 重新得到 12 个 128×64的矩阵
       8、然后把这12个 128×64矩阵合并回128×768
       9、这里少了一步， QKV计算完成后， 还有个收尾的线性层
       二、残差（1） + ln（注意 ln也有学习参数 两个  γ 和 β）
       拿着 多头自注意力的返回值 和 他的入参 进行 矩阵相加 ，然后进行 ln  layernorm 行归一化
       三、feed-forward network 前馈网络层
       f(x) = gelu(xw1 + b1)w2 + b2
       w1 = 768 × 4 =  768 × 3072
       b1 = 128 × 3072
       gelu激活函数 不改变形状
       w1x是将我们的 128×768 放大到 128 × 3072
       b1 也是 128×3072
       然后经过gelu激活函数， 计算一下
       然后在通过w2进行缩小
       w2 = 3072 × 768 计算完成后 形状回到了 128×768， b2也是这个形状
       四、残差（2）（注意 ln也有学习参数 两个  γ 和 β）

       需要从bert的model拿的参数有哪些，才能进行diy
       线性QKV3个矩阵的 w1 b1  w2  b2   w3  b3  6个参数
       qkv的计算结果还需要经过一层线性层  w4 b4 2个参数
       残差1的归一化两个参数 γ1  β1
       前馈层 w1  b1  w2  b2 4个参数
       残差2的归一化两个参数 γ2 β2
       一共 6 + 2 + 2 + 4 + 2 = 16个参数
    """
    # 准备multi-head self-attention的参数
    # QKV3个 + 收尾的1个线性层的wb参数，一共8个
    head_w1 = weight_dict.get("attention.self.query.weight")
    head_b1 = weight_dict.get("attention.self.query.bias")
    head_w2 = weight_dict.get("attention.self.key.weight")
    head_b2 = weight_dict.get("attention.self.key.bias")
    head_w3 = weight_dict.get("attention.self.value.weight")
    head_b3 = weight_dict.get("attention.self.value.bias")
    head_w4 = weight_dict.get("attention.output.dense.weight")
    head_b4 = weight_dict.get("attention.output.dense.bias")
    
    # 归一化1
    gama1 = weight_dict.get("attention.output.LayerNorm.weight")
    beta1 = weight_dict.get("attention.output.LayerNorm.bias")
    
    # 前馈层，4个参数
    feed_w1 = weight_dict.get("intermediate.dense.weight")
    feed_b1 = weight_dict.get("intermediate.dense.bias")
    feed_w2 = weight_dict.get("output.dense.weight")
    feed_b2 = weight_dict.get("output.dense.bias")
    
    # 归一化2
    gama2 = weight_dict.get("output.LayerNorm.weight")
    beta2 = weight_dict.get("output.LayerNorm.bias")
    
    head_count = 12
    
    # QKV首先用input计算一下，input = (128, 768)
    q_result = input @ head_w1.transpose(0, 1) + head_b1
    k_result = input @ head_w2.transpose(0, 1) + head_b2
    v_result = input @ head_w3.transpose(0, 1) + head_b3
    # 然后把 q_result  k_result v_result ，每个切分12份
    q_split_list = torch.chunk(q_result, head_count, axis=1)
    k_split_list = torch.chunk(k_result, head_count, axis=1)
    v_split_list = torch.chunk(v_result, head_count, axis=1)
    
    """
    然后把q_split_list  和 k_split_list
    每个元素，  
    1、q @ k的转置
    2、结果/8
    3、每一行softmax
    """
    q_k_v_result_list = []
    for index, q_split_item in enumerate(q_split_list):
        k_split_item = k_split_list[index]
        v_split_item = v_split_list[index]
        # q@k的转置   128 × 64 @ 64 × 128 = 128 × 128
        q_k_result = q_split_item @ k_split_item.transpose(-1, -2)
        # 结果/8
        q_k_result = q_k_result / math.sqrt(768 / head_count)
        # 每一行softmax
        q_k_result = torch.softmax(q_k_result, dim=-1)
        # q_k_result @ v_split_item, 128 × 128 @ 128 × 64 = 128 × 64
        q_k_v_result = q_k_result @ v_split_item
        q_k_v_result_list.append(q_k_v_result)
    # 将12个   128×64在1维合并到一起， 重新变成 128×768
    q_k_v_result_cat = torch.cat(q_k_v_result_list, dim=1)
    # 在过一次线性层 w4x + b4
    multi_head_final_result = q_k_v_result_cat @ head_w4.transpose(0, 1) + head_b4
    
    # 二、残差 + 归一化
    # ln1 = nn.LayerNorm(768, eps=1e-12)
    # ln1.weight = nn.Parameter(gama1)
    # ln1.bias = nn.Parameter(beta1)
    # 计算前后 形状都是 128 × 768 
    residual1 = input + multi_head_final_result
    # 行归一化
    # residual1_norm_result = ln1(residual1)
    residual1_norm_result = layer_norm(residual1, gama1, beta1, eps=1e-12)

    
    # 前馈处理f(x) = gelu(xw1 + b1)w2 + b2
    # 首先计算内层xw1 + b1
    feed_result = residual1_norm_result @ feed_w1.transpose(0, 1) + feed_b1
    gelu = nn.GELU()
    feed_result = gelu(feed_result) @ feed_w2.transpose(0, 1) + feed_b2
    
    # 残差2 + 归一化
    residual2 = residual1_norm_result + feed_result
    # 行归一化
    # ln2 = nn.LayerNorm(768, eps=1e-12)
    # ln2.weight = nn.Parameter(gama2)
    # ln2.bias = nn.Parameter(beta2)
    final_result = layer_norm(residual2, gama2, beta2, eps=1e-12)
    
    return final_result
    
    
def layer_norm(x, weight, bias, eps=1e-12):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    return weight * (x - mean) / torch.sqrt(var + eps) + bias
# ===================== 官方Transformer层 =====================
def get_official_transformer_layer(weights):
    """
    用官方API构建BERT的单层Transformer，加载相同权重
    """
    config = BertConfig(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        layer_norm_eps=1e-12  # BERT默认参数
    )
    # 官方BERT的单层Transformer
    official_layer = BertModel(config).encoder.layer[0]
    
    # 加载权重到官方层
    official_layer.load_state_dict(weights)
    official_layer.eval()
    return official_layer

# ===================== 核心：对比验证 =====================
def verify():
    # 1. 加载本地 safetensors 权重（修改为你的文件路径）
    all_weights = load_file("/Users/yushichang/python/hub-kzSW/于世昌/week04/model.safetensors")  # 👈 改成你的路径！
    
    # 2. 提取【BERT第一层Transformer】的所有权重（关键！）
    layer_prefix = "bert.encoder.layer.0."  # 第0层，你可以改成1/2...验证任意层
    layer_weights = {}
    for k, v in all_weights.items():
        if k.startswith(layer_prefix):
            new_k = k.replace(layer_prefix, "")  # 👈 必须加这行
            layer_weights[new_k] = v
    
    # 把 gamma/beta 改成官方 LayerNorm 要的 weight/bias
    if "attention.output.LayerNorm.gamma" in layer_weights:
        layer_weights["attention.output.LayerNorm.weight"] = layer_weights.pop("attention.output.LayerNorm.gamma")
    if "attention.output.LayerNorm.beta" in layer_weights:
        layer_weights["attention.output.LayerNorm.bias"] = layer_weights.pop("attention.output.LayerNorm.beta")

    if "output.LayerNorm.gamma" in layer_weights:
        layer_weights["output.LayerNorm.weight"] = layer_weights.pop("output.LayerNorm.gamma")
    if "output.LayerNorm.beta" in layer_weights:
        layer_weights["output.LayerNorm.bias"] = layer_weights.pop("output.LayerNorm.beta")

    # 3. 构造固定输入（必须用固定随机数，保证两次输入完全一致）
    torch.manual_seed(42)
    input_mat = torch.rand(128, 768)  # 和你一样的输入形状

    # 4. 计算：你的输出 + 官方输出
    diy_output = bert_transformer_diy_weight(input_mat, layer_weights)
    official_layer = get_official_transformer_layer(layer_weights)
    official_output = official_layer(input_mat.unsqueeze(0))[0].squeeze(0)  # 适配维度

    # 5. 计算误差（核心验证标准！）
    abs_error = torch.abs(diy_output - official_output).max()
    print(f"✅ 最大绝对误差: {abs_error.item()}")
    print(f"✅ 误差 < 1e-5 说明你的手写Transformer完全正确！")

def show_all_model_params():
    # all_weights = load_file("/Users/yushichang/python/hub-kzSW/于世昌/week04/model.safetensors")  # 👈 改成你的路径！
    # for k, v in all_weights.items():
    #     if k.startswith("bert.encoder.layer.0"):
    #         print(f"{k}: {v}")
    verify()

def main():
    show_all_model_params()
if __name__ == "__main__":
    main()