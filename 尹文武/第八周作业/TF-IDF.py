# TF-IDF 算法实现

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

def compute_tf_idf(document, corpus):
    """计算 TF-IDF"""
    tf = compute_tf(document)
    idf = compute_idf(corpus)
    tf_idf = {word: tf[word] * idf.get(word, 0) for word in tf}
    return tf_idf


# 引入三方库做效果对比
from sklearn.feature_extraction.text import TfidfVectorizer
# 测试代码
if __name__ == "__main__":
    corpus = [
        "the cat sat on the mat".split(),
        "the dog sat on the log".split(),
        "the cat and the dog are friends".split()
    ]
    document = "the cat sat on the mat".split()
    tf_idf = compute_tf_idf(document, corpus)
    print(tf_idf)

    # 使用 sklearn 计算 TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([' '.join(doc) for doc in corpus])
    feature_names = vectorizer.get_feature_names_out()
    tf_idf_sklearn = dict(zip(feature_names, X[0].toarray()[0]))
    print(tf_idf_sklearn)
