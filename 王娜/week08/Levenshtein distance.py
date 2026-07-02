"""
Levenshtein Distance（编辑距离）

传统的文本匹配方法，衡量两个字符串之间的相似程度：
- 通过计算将一个字符串转换成另一个字符串所需的最少编辑操作次数
- 允许的编辑操作：插入、删除、替换

例如：
    "kitten" -> "sitting"
    1. kitten -> sitten (替换 k 为 s)
    2. sitten -> sittin (替换 e 为 i)
    3. sittin -> sitting (插入 g)
    编辑距离 = 3

相似度可以表示为：
    similarity = 1 - distance / max(len(s1), len(s2))
"""

from functools import lru_cache


def levenshtein_distance_2d(s1: str, s2: str) -> int:
    """
    标准动态规划实现（二维矩阵）。

    时间复杂度: O(m * n)
    空间复杂度: O(m * n)
    """
    m, n = len(s1), len(s2)

    # dp[i][j] 表示 s1[:i] 与 s2[:j] 的编辑距离
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化边界：空字符串转换为目标字符串
    for i in range(m + 1):
        dp[i][0] = i  # 删除 i 个字符
    for j in range(n + 1):
        dp[0][j] = j  # 插入 j 个字符

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 删除 s1[i-1]
                dp[i][j - 1] + 1,      # 插入 s2[j-1]
                dp[i - 1][j - 1] + cost # 替换或不操作
            )

    return dp[m][n]


def levenshtein_distance_optimized(s1: str, s2: str) -> int:
    """
    空间优化版本：只使用两行（当前行和上一行）。

    时间复杂度: O(m * n)
    空间复杂度: O(min(m, n))
    """
    # 让 s2 是较短的字符串，减少空间占用
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)

    prev = list(range(n + 1))  # 上一行 dp[i-1][*]
    curr = [0] * (n + 1)       # 当前行 dp[i][*]

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1

            curr[j] = min(
                prev[j] + 1,       # 删除
                curr[j - 1] + 1,   # 插入
                prev[j - 1] + cost # 替换/匹配
            )
        prev, curr = curr, prev  # 交换引用，复用列表

    return prev[n]


@lru_cache(maxsize=None)
def levenshtein_distance_recursive(s1: str, s2: str) -> int:
    """
    递归 + 记忆化实现，更易理解但效率较低。
    """
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    if s1[-1] == s2[-1]:
        cost = 0
    else:
        cost = 1

    return min(
        levenshtein_distance_recursive(s1[:-1], s2) + 1,       # 删除
        levenshtein_distance_recursive(s1, s2[:-1]) + 1,       # 插入
        levenshtein_distance_recursive(s1[:-1], s2[:-1]) + cost # 替换
    )


def similarity(s1: str, s2: str) -> float:
    """
    基于编辑距离的归一化相似度，范围 [0, 1]。
    """
    if len(s1) == 0 and len(s2) == 0:
        return 1.0

    distance = levenshtein_distance_optimized(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1 - distance / max_len


def demo():
    """示例：比较几组中文/英文句子的编辑距离。"""
    pairs = [
        ("kitten", "sitting"),
        ("sunday", "saturday"),
        ("你好世界", "你好世介"),
        ("今天天气很好", "今天天气真好"),
        ("自然语言处理", "自然语音处理"),
        ("", "abc"),
        ("abc", "abc"),
    ]

    print(f"{'s1':<15} {'s2':<15} {'distance':<10} {'similarity':<12}")
    print("-" * 55)

    for s1, s2 in pairs:
        dist = levenshtein_distance_optimized(s1, s2)
        sim = similarity(s1, s2)
        print(f"{s1:<15} {s2:<15} {dist:<10} {sim:<12.4f}")


if __name__ == "__main__":
    demo()

    # 单元测试：三种实现结果应一致
    test_cases = [
        ("kitten", "sitting"),
        ("", ""),
        ("a", "b"),
        ("abc", "abc"),
        ("自然语言", "自然言语"),
    ]

    print("\n一致性检查：")
    for s1, s2 in test_cases:
        d1 = levenshtein_distance_2d(s1, s2)
        d2 = levenshtein_distance_optimized(s1, s2)
        d3 = levenshtein_distance_recursive(s1, s2)
        assert d1 == d2 == d3, f"结果不一致: {s1}, {s2}"
        print(f"  {s1!r} vs {s2!r}: {d1} OK")

    print("\n所有测试通过！")
