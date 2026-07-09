"""
传统文本匹配方法

教学重点：
  1. 编辑距离（Levenshtein Distance）— 基于字符序列的相似度
  2. 词向量（Word2Vec）— 基于词嵌入的语义相似度
  3. TF-IDF — 基于词频统计的相似度
  4. BM25 — 改进的TF-IDF，考虑文档长度归一化

使用方式：
  from model import EditDistanceModel, WordVectorModel, TFIDFModel, BM25Model

依赖：
  pip install gensim scikit-learn jieba
"""

import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba


# ── 编辑距离模型 ──────────────────────────────────────────────────────────

class EditDistanceModel:
    """
    编辑距离（Levenshtein Distance）

    原理：
      计算两个字符串之间的最小编辑操作数（插入、删除、替换）
      转换为相似度：sim = 1 - distance / max(len(s1), len(s2))

    教学价值：
      - 最基础的字符串匹配方法，不考虑语义
      - 对字符顺序敏感，适合检测拼写错误或微小差异
      - 不适合语义相似度判断（如"喜欢打篮球" vs "爱打篮球"）

    优点：
      - 计算简单快速
      - 无需训练数据

    缺点：
      - 只考虑字符表面差异
      - 无法处理同义词、语义相似
    """

    def __init__(self):
        pass

    def edit_distance(self, s1, s2):
        """计算编辑距离"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,    # 删除
                        dp[i][j-1] + 1,    # 插入
                        dp[i-1][j-1] + 1   # 替换
                    )

        return dp[m][n]

    def similarity(self, s1, s2):
        """计算相似度分数 [0, 1]"""
        if len(s1) == 0 and len(s2) == 0:
            return 1.0
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        distance = self.edit_distance(s1, s2)
        return 1.0 - distance / max_len

    def predict(self, sentence_pairs):
        """批量预测相似度"""
        similarities = []
        for s1, s2 in sentence_pairs:
            sim = self.similarity(s1, s2)
            similarities.append(sim)
        return np.array(similarities)


# ── 词向量模型 ────────────────────────────────────────────────────────────

class WordVectorModel:
    """
    词向量（Word2Vec）语义相似度

    原理：
      1. 使用预训练词向量（如腾讯词向量）
      2. 对句子进行分词，获取每个词的向量
      3. 计算句子向量（词向量平均）
      4. 计算余弦相似度

    教学价值：
      - 从字符匹配到语义匹配的关键进步
      - 词向量能捕捉语义相似性（如"喜欢"和"爱"）
      - 但平均向量会丢失词序信息

    优点：
      - 考虑语义相似度
      - 可以处理同义词

    缺点：
      - 平均向量丢失词序信息
      - 需要预训练词向量
      - OOV（未登录词）问题
    """

    def __init__(self, word2vec_model=None):
        """
        参数：
          word2vec_model : gensim KeyedVectors 对象
                          如果为None，使用随机向量演示
        """
        self.word2vec_model = word2vec_model
        self.vector_size = 300 if word2vec_model is None else word2vec_model.vector_size

    def tokenize(self, text):
        """使用jieba分词"""
        return list(jieba.cut(text))

    def get_sentence_vector(self, text):
        """计算句子向量（词向量平均）"""
        words = self.tokenize(text)
        vectors = []

        for word in words:
            if self.word2vec_model is not None and word in self.word2vec_model:
                vectors.append(self.word2vec_model[word])
            else:
                # OOV或演示模式：使用随机向量
                vectors.append(np.random.randn(self.vector_size))

        if len(vectors) == 0:
            return np.zeros(self.vector_size)

        return np.mean(vectors, axis=0)

    def cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def similarity(self, s1, s2):
        """计算两个句子的相似度"""
        vec1 = self.get_sentence_vector(s1)
        vec2 = self.get_sentence_vector(s2)
        return self.cosine_similarity(vec1, vec2)

    def predict(self, sentence_pairs):
        """批量预测相似度"""
        similarities = []
        for s1, s2 in sentence_pairs:
            sim = self.similarity(s1, s2)
            similarities.append(sim)
        return np.array(similarities)


# ── TF-IDF 模型 ───────────────────────────────────────────────────────────

class TFIDFModel:
    """
    TF-IDF（Term Frequency-Inverse Document Frequency）

    原理：
      1. TF：词在文档中的频率
      2. IDF：词在所有文档中的稀有程度（log(N/df))
      3. TF-IDF = TF * IDF，衡量词的重要性
      4. 计算两句话的TF-IDF向量余弦相似度

    教学价值：
      - 统计方法，考虑词的重要性
      - IDF能降低常见词（如"的"、"是"）的权重
      - 适合关键词匹配场景

    优点：
      - 不需要预训练模型
      - 考虑词的重要性（IDF）
      - 计算速度快

    缺点：
      - 不考虑语义相似性
      - 不考虑词序
      - 稀疏向量，维度高
    """

    def __init__(self):
        self.vectorizer = None
        self.is_fitted = False

    def fit(self, sentences):
        """训练TF-IDF模型"""
        # 使用jieba分词后的文本
        tokenized = [' '.join(jieba.cut(s)) for s in sentences]
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(tokenized)
        self.is_fitted = True

    def transform(self, sentences):
        """将句子转换为TF-IDF向量"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()")
        tokenized = [' '.join(jieba.cut(s)) for s in sentences]
        return self.vectorizer.transform(tokenized)

    def similarity(self, s1, s2):
        """计算两个句子的相似度"""
        vec1 = self.transform([s1])
        vec2 = self.transform([s2])
        # 稀疏矩阵的余弦相似度
        dot_product = vec1.dot(vec2.T).toarray()[0][0]
        norm1 = np.sqrt(vec1.dot(vec1.T).toarray()[0][0])
        norm2 = np.sqrt(vec2.dot(vec2.T).toarray()[0][0])
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def predict(self, sentence_pairs):
        """批量预测相似度"""
        similarities = []
        for s1, s2 in sentence_pairs:
            sim = self.similarity(s1, s2)
            similarities.append(sim)
        return np.array(similarities)


# ── BM25 模型 ─────────────────────────────────────────────────────────────

class BM25Model:
    """
    BM25（Best Matching 25）

    原理：
      BM25是TF-IDF的改进版本，主要改进：
      1. TF饱和：使用k1参数控制TF的增长上限，避免高频词过度影响
         TF_component = (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl/avdl))
      2. 文档长度归一化：使用b参数控制文档长度的影响
      3. IDF改进：使用平滑的IDF公式

      BM25公式：
        score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))

      其中：
        - f(qi, D)：词qi在文档D中的频率
        - |D|：文档D的长度
        - avgdl：平均文档长度
        - k1：通常取1.2-2.0，控制TF饱和
        - b：通常取0.75，控制长度归一化

    教学价值：
      - BM25是搜索引擎的标准排序算法
      - 相比TF-IDF，BM25更适合长短不一的文档
      - TF饱和避免了高频词的过度影响
      - 文档长度归一化平衡了长短文档

    优点：
      - 相比TF-IDF更合理
      - 适合文档检索场景
      - 考虑文档长度影响

    缺点：
      - 不考虑语义相似性
      - 参数需要调优
    """

    def __init__(self, k1=1.5, b=0.75):
        """
        参数：
          k1 : TF饱和参数，通常1.2-2.0
          b  : 文档长度归一化参数，通常0.75
        """
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}      # 词的文档频率
        self.doc_lens = []       # 文档长度列表
        self.avgdl = 0           # 平均文档长度
        self.doc_term_freqs = [] # 每个文档的词频
        self.N = 0               # 文档总数
        self.idf = {}            # IDF值

    def tokenize(self, text):
        """使用jieba分词"""
        return list(jieba.cut(text))

    def fit(self, documents):
        """训练BM25模型"""
        self.N = len(documents)
        self.doc_lens = []
        self.doc_term_freqs = []
        self.doc_freqs = {}

        # 统计每个文档的词频和长度
        for doc in documents:
            tokens = self.tokenize(doc)
            self.doc_lens.append(len(tokens))

            term_freq = Counter(tokens)
            self.doc_term_freqs.append(term_freq)

            # 更新文档频率
            for term in term_freq.keys():
                if term not in self.doc_freqs:
                    self.doc_freqs[term] = 0
                self.doc_freqs[term] += 1

        # 计算平均文档长度
        self.avgdl = np.mean(self.doc_lens) if self.doc_lens else 0

        # 计算IDF
        for term, df in self.doc_freqs.items():
            # BM25 IDF公式（带平滑）
            self.idf[term] = np.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query, doc_idx):
        """计算query与第doc_idx个文档的BM25分数"""
        query_tokens = self.tokenize(query)
        doc_term_freq = self.doc_term_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]

        score = 0.0
        for term in query_tokens:
            if term not in doc_term_freq:
                continue

            f = doc_term_freq[term]
            idf = self.idf.get(term, 0)

            # BM25 TF分量
            tf_component = (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))

            score += idf * tf_component

        return score

    def similarity(self, query, document):
        """
        计算query和document的相似度
        注意：BM25原本是query对document的评分，这里简化处理
        """
        # 将document加入文档库，计算query对它的BM25分数
        tokens = self.tokenize(document)
        doc_len = len(tokens)
        term_freq = Counter(tokens)

        score = 0.0
        query_tokens = self.tokenize(query)

        for term in query_tokens:
            if term not in term_freq:
                continue

            f = term_freq[term]
            idf = self.idf.get(term, np.log((self.N + 0.5) / 0.5 + 1))

            # BM25 TF分量
            tf_component = (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))

            score += idf * tf_component

        # 归一化到[0, 1]范围（简化处理）
        return max(0, min(1, score / 10))

    def predict(self, sentence_pairs):
        """批量预测相似度"""
        similarities = []
        for s1, s2 in sentence_pairs:
            sim = self.similarity(s1, s2)
            similarities.append(sim)
        return np.array(similarities)


# ── 工厂函数 ──────────────────────────────────────────────────────────────

def build_edit_distance_model():
    """构建编辑距离模型"""
    return EditDistanceModel()


def build_word_vector_model(word2vec_model=None):
    """构建词向量模型"""
    return WordVectorModel(word2vec_model)


def build_tfidf_model(sentences=None):
    """
    构建TF-IDF模型

    参数：
      sentences : 用于训练的句子列表，如果为None则返回未训练的模型
    """
    model = TFIDFModel()
    if sentences is not None:
        model.fit(sentences)
    return model


def build_bm25_model(documents=None, k1=1.5, b=0.75):
    """
    构建BM25模型

    参数：
      documents : 用于训练的文档列表，如果为None则返回未训练的模型
      k1        : TF饱和参数
      b         : 文档长度归一化参数
    """
    model = BM25Model(k1=k1, b=b)
    if documents is not None:
        model.fit(documents)
    return model