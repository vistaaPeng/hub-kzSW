from transformers import BertModel
import numpy as np
import torch
import math

# 测试 bert 模型
bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False, num_hidden_layers=1)
bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False)

bert.eval()

x = np.array([456,798,132,1654])
x_tensor = torch.LongTensor([x])

seqence_output, pooler_output = bert(x_tensor)

print(seqence_output.shape, pooler_output.shape)

# 查看模型配置文件
config = bert.config
print(config)

# 查看模型参数
state_dict = bert.state_dict()
print(len(state_dict))
print(state_dict.keys())

#softmax归一化
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

#gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

# 手工计算 bert 模型
class DiyBert():
    
    def __init__(self, config, state_dict):
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        # embeddings
        self.word_embeddings = state_dict['embeddings.word_embeddings.weight'].numpy()
        self.position_embeddings = state_dict['embeddings.position_embeddings.weight'].numpy()
        self.token_type_embeddings = state_dict['embeddings.token_type_embeddings.weight'].numpy()
        self.embeddings_layer_norm_w, self.embeddings_layer_norm_b = state_dict['embeddings.LayerNorm.weight'].numpy(), state_dict['embeddings.LayerNorm.bias'].numpy()
        # tarnsformer
        self.transformer_weights = []
        for i in range(self.num_hidden_layers):
            q_w, q_b = state_dict[f'encoder.layer.{i}.attention.self.query.weight'].numpy(), state_dict[f'encoder.layer.{i}.attention.self.query.bias'].numpy()
            k_w, k_b = state_dict[f'encoder.layer.{i}.attention.self.key.weight'].numpy(), state_dict[f'encoder.layer.{i}.attention.self.key.bias'].numpy()
            v_w, v_b = state_dict[f'encoder.layer.{i}.attention.self.value.weight'].numpy(), state_dict[f'encoder.layer.{i}.attention.self.value.bias'].numpy()
            attention_output_w, attention_output_b = state_dict[f'encoder.layer.{i}.attention.output.dense.weight'].numpy(), state_dict[f'encoder.layer.{i}.attention.output.dense.bias'].numpy()
            attention_layer_norm_w, attention_layer_norm_b = state_dict[f'encoder.layer.{i}.attention.output.LayerNorm.weight'].numpy(), state_dict[f'encoder.layer.{i}.attention.output.LayerNorm.bias'].numpy()
            ffn_intermediate_w, ffn_intermediate_b = state_dict[f'encoder.layer.{i}.intermediate.dense.weight'].numpy(), state_dict[f'encoder.layer.{i}.intermediate.dense.bias'].numpy()
            ffn_output_w, ffn_output_b = state_dict[f'encoder.layer.{i}.output.dense.weight'].numpy(), state_dict[f'encoder.layer.{i}.output.dense.bias'].numpy()
            ffn_layer_norm_w, ffn_layer_norm_b = state_dict[f'encoder.layer.{i}.output.LayerNorm.weight'].numpy(), state_dict[f'encoder.layer.{i}.output.LayerNorm.bias'].numpy()
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_w, attention_output_b, attention_layer_norm_w, attention_layer_norm_b, 
                                             ffn_intermediate_w, ffn_intermediate_b, ffn_output_w, ffn_output_b, ffn_layer_norm_w, ffn_layer_norm_b])
        self.pooler_dense_w, self.pooler_dense_b = state_dict['pooler.dense.weight'].numpy(), state_dict['pooler.dense.bias'].numpy()

    def forward(self, x):
        x = self.emmbedding_forward(x)
        seqence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(seqence_output[0])
        return seqence_output, pooler_output

    def emmbedding_forward(self, x):
        we = self.get_embedding(self.word_embeddings, x)
        pe = self.get_embedding(self.position_embeddings, np.array(list(range(len(x)))))
        te = self.get_embedding(self.token_type_embeddings, np.array([0] * len(x)))
        e = we + pe + te
        e = self.layer_norm(e, self.embeddings_layer_norm_w, self.embeddings_layer_norm_b)
        return e
        
    def get_embedding(self, embedding, index):
        return embedding[index]
    
    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x
    
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_hidden_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x

    def single_transformer_layer_forward(self, x, layer_index):
        q_w, q_b, k_w, k_b, v_w, v_b, attention_output_w, attention_output_b, attention_layer_norm_w, attention_layer_norm_b, \
        ffn_intermediate_w, ffn_intermediate_b, ffn_output_w, ffn_output_b, ffn_layer_norm_w, ffn_layer_norm_b = self.transformer_weights[layer_index]
        attention_output = self.attention_forward(x, q_w, q_b, k_w, k_b, v_w, v_b, attention_output_w, attention_output_b, self.num_attention_heads, self.hidden_size)
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)
        ffn_output = self.ffn_forward(x, ffn_intermediate_w, ffn_intermediate_b, ffn_output_w, ffn_output_b)
        x = self.layer_norm(x + ffn_output, ffn_layer_norm_w, ffn_layer_norm_b)
        return x
    
    def attention_forward(self, x, q_w, q_b, k_w, k_b, v_w, v_b, attention_output_w, attention_output_b, num_attention_heads, hidden_size):
        q = x @ q_w.T + q_b
        k = x @ k_w.T + k_b
        v = x @ v_w.T + v_b
        attention_head_size = int(hidden_size / num_attention_heads)
        q = self.transpose_multi_head(q, attention_head_size, num_attention_heads)
        k = self.transpose_multi_head(k, attention_head_size, num_attention_heads)
        v = self.transpose_multi_head(v, attention_head_size, num_attention_heads)
        qk = q @ k.swapaxes(1, 2)
        qk = qk / np.sqrt(attention_head_size)
        qk = softmax(qk)
        qkv = qk @ v    
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        attention_output = qkv @ attention_output_w.T + attention_output_b
        return attention_output

    def transpose_multi_head(self, x, attention_head_size, num_attention_heads):
        max_len, _ = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(1, 0)
        return x
    
    def ffn_forward(self, x, ffn_intermediate_w, ffn_intermediate_b, ffn_output_w, ffn_output_b):
        ffn_intermediate = x @ ffn_intermediate_w.T + ffn_intermediate_b
        ffn_intermediate = gelu(ffn_intermediate)
        ffn_output = ffn_intermediate @ ffn_output_w.T + ffn_output_b
        return ffn_output

    def pooler_output_layer(self, x):
        x = x @ self.pooler_dense_w.T + self.pooler_dense_b
        x = np.tanh(x)
        return pooler_output


diy_bert = DiyBert(config, state_dict)
diy_sequence_output, diy_pooler_output = diy_bert.forward(x)