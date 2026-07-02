"""
文本匹配项目

包含传统方法和深度学习方法的对比：
  - 编辑距离（Edit Distance）
  - 词向量（Word Vector）
  - TF-IDF
  - BM25
  - BiEncoder（表示型）
  - CrossEncoder（交互型）
"""

from .dataset import (
    PairDataset,
    TripletDataset,
    CrossEncoderDataset,
    TraditionalDataset,
    build_pair_loaders,
    build_triplet_loader,
    build_crossencoder_loaders,
    build_traditional_loaders,
)

from .model import (
    EditDistanceModel,
    WordVectorModel,
    TFIDFModel,
    BM25Model,
    build_edit_distance_model,
    build_word_vector_model,
    build_tfidf_model,
    build_bm25_model,
)

from .biencoder import (
    BiEncoder,
    build_biencoder,
)

from .crossencoder import (
    CrossEncoder,
    build_crossencoder,
)

from .evaluate import (
    eval_biencoder,
    eval_crossencoder,
    eval_traditional,
    plot_similarity_distribution,
)