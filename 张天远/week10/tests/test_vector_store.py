"""
测试向量存储 —— Embedding 编码 + FAISS 索引 + 检索。

注意：test_model_loading 和 test_end_to_end 需要下载模型，
首次运行较慢（bge-base-zh-v1.5 ~363MB）。
"""
import pytest
from src.chunkers.narrative import Chunk
from src.retrievers.vector_store import VectorStore


@pytest.fixture
def sample_chunks():
    """创建测试用的 chunk 列表。"""
    return [
        Chunk(
            chunk_id="test_0000",
            text="Rust 的所有权系统确保内存安全，无需垃圾回收器。",
            metadata={"source_url": "url1", "source_name": "book"},
        ),
        Chunk(
            chunk_id="test_0001",
            text="借用（Borrowing）允许在不转移所有权的情况下使用值。",
            metadata={"source_url": "url1", "source_name": "book"},
        ),
        Chunk(
            chunk_id="test_0002",
            text="生命周期（Lifetime）确保引用始终有效。",
            metadata={"source_url": "url2", "source_name": "book"},
        ),
        Chunk(
            chunk_id="test_0003",
            text="Python 的变量不需要声明类型，是动态类型语言。",
            metadata={"source_url": "url3", "source_name": "reference"},
        ),
    ]


class TestVectorStore:
    """向量存储测试"""

    def test_model_loading(self):
        """模型加载成功，维度正确（768）"""
        store = VectorStore()
        assert store.dim == 768
        assert store.model is not None

    def test_encode_single_text(self):
        """单条文本编码"""
        store = VectorStore()
        vec = store.encode(["Rust 所有权"])
        assert vec.shape == (1, 768)

    def test_encode_batch(self):
        """批量编码"""
        store = VectorStore()
        texts = ["Rust 所有权", "借用规则", "生命周期"]
        vec = store.encode(texts)
        assert vec.shape == (3, 768)

    def test_build_and_search(self, sample_chunks):
        """构建索引 + 检索"""
        store = VectorStore()
        store.build_index(sample_chunks)

        # 精确查询 "所有权"
        results = store.search("所有权是什么", top_k=3)
        assert len(results) == 3
        # 第一结果应该关于所有权（不是 Python）
        top_chunk, top_score = results[0]
        assert "所有权" in top_chunk.text or "own" in top_chunk.text.lower()

    def test_save_and_load_index(self, sample_chunks, tmp_path):
        """索引持久化：保存 + 重新加载后检索结果一致"""
        store = VectorStore()
        store.build_index(sample_chunks)

        index_path = str(tmp_path / "test_index.faiss")
        chunks_path = str(tmp_path / "test_chunks.json")

        store.save(index_path, chunks_path)

        # 重新加载
        store2 = VectorStore()
        store2.load(index_path, chunks_path)

        # 两次检索结果一致
        r1 = store.search("Rust 所有权", top_k=3)
        r2 = store2.search("Rust 所有权", top_k=3)
        assert [c.chunk_id for c, _ in r1] == [c.chunk_id for c, _ in r2]
