# BM25 算法实现

import math
from collections import Counter
def compute_tf(document):
    """计算词频（TF）"""
    word_count = Counter(document)
    total_words = len(document)
    tf = {word: count / total_words for word, count in word_count.items()}
    return tf

def compute_idf(corpus):
    """计算逆文档频率（IDF）"""
    num_documents = len(corpus)
    idf = {}
    for document in corpus:
        unique_words = set(document)
        for word in unique_words:
            idf[word] = idf.get(word, 0) + 1
    idf = {word: math.log(num_documents / count) for word, count in idf.items()}
    return idf

def compute_bm25(document, corpus, k1=1.5, b=0.75):
    """计算 BM25"""
    tf = compute_tf(document)
    idf = compute_idf(corpus)
    avg_doc_len = sum(len(doc) for doc in corpus) / len(corpus)
    bm25 = {}
    for word in tf:
        term_freq = tf[word]
        doc_freq = idf.get(word, 0)
        bm25[word] = idf.get(word, 0) * (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * len(document) / avg_doc_len))
    return bm25

# 测试代码
if __name__ == "__main__":
    corpus = [
        "the cat sat on the mat".split(),
        "the dog sat on the log".split(),
        "the cat and the dog are friends".split()
    ]
    document = "the cat sat on the mat".split()
    bm25 = compute_bm25(document, corpus)
    print(bm25)
