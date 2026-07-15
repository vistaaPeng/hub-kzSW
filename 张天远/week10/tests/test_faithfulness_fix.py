"""
测试 Faithfulness 分层评估改动
  1. 问题分类器正确性
  2. 10 道低分题的类型判定
  3. 推导类 vs 事实类的 prompt 差异

用法: python tests/test_faithfulness_fix.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import RAGEvaluator

# 10 道低分题（来自 eval_20260706_175620）
LOW_FAITH_QUESTIONS = {
    "m24": "Rust 中 Clone 和 Copy trait 的区别是什么？哪些类型实现了 Copy？",
    "m26": "Rust 中什么是函数指针？fn 类型和 Fn trait 有什么区别？",
    "m36": "Rust 中什么是孤儿规则（orphan rule）？它如何限制 trait 实现？",
    "m38": "Rust 中什么是关联类型（associated type）？它和泛型参数有什么区别？",
    "m25": "Rust 中如何实现多态？trait 对象（dyn Trait）和泛型各有什么优劣？",
    "m10": "Rust 中什么是模块系统（module system）？mod、use、pub 关键字的作用是什么？",
    "m19": "Rust 中如何编写文档注释？rustdoc 工具如何生成文档？",
    "m14": "Rust 中什么是闭包（closure）？闭包如何捕获环境变量？",
    "m35": "Rust 中 #[derive] 属性可以自动实现哪些 trait？如何自定义 derive？",
    "m22": "Rust 中宏（macro）如何使用？macro_rules! 和过程宏的区别是什么？",
}

EXPECTED_TYPES = {
    "m24": "inferential",   # "区别"
    "m26": "inferential",   # "区别"
    "m36": "factual",       # 概念 + 机制（无对比关键词）
    "m38": "inferential",   # "区别"
    "m25": "inferential",   # "优劣"
    "m10": "factual",       # 概念解释
    "m19": "factual",       # 用法
    "m14": "factual",       # 概念+机制
    "m35": "factual",       # 用法
    "m22": "inferential",   # "区别"
}

def test_classifier():
    """测试问题分类器。"""
    evaluator = RAGEvaluator.__new__(RAGEvaluator)
    passed = 0
    failed = 0
    for tid, question in LOW_FAITH_QUESTIONS.items():
        result = evaluator._classify_question_type(question)
        expected = EXPECTED_TYPES[tid]
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} {tid}: {result:12s} (expected {expected})")
        print(f"   Q: {question[:80]}")
    print(f"\n{passed}/{passed+failed} passed, {failed} failed")
    return failed == 0


def test_prompt_difference():
    """验证推导类和事实类使用不同 prompt。"""
    evaluator = RAGEvaluator.__new__(RAGEvaluator)
    
    fact_q = "Rust 中什么是所有权？"
    infer_q = "Rust 中所有权和借用的区别是什么？"
    
    t1 = evaluator._classify_question_type(fact_q)
    t2 = evaluator._classify_question_type(infer_q)
    
    print(f"\n所有权是什么 → {t1}")
    print(f"所有权和借用的区别 → {t2}")
    
    assert t1 == "factual", f"Expected factual, got {t1}"
    assert t2 == "inferential", f"Expected inferential, got {t2}"
    print("✅ Prompt 类型区分正确")


if __name__ == "__main__":
    print("=" * 60)
    print("Faithfulness 分层评估改动测试")
    print("=" * 60)
    
    test_prompt_difference()
    print()
    ok = test_classifier()
    
    if ok:
        print("\n🎉 全部通过！")
    else:
        print("\n❌ 存在失败，请检查分类器逻辑")
        sys.exit(1)
