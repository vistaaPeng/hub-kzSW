# 代码实现编辑距离算法
def edit_distance(str1, str2):
    # str1 和 str2 是待比较的两个字符串。
    # m 和 n 分别是这两个字符串的长度。这是构建动态规划（DP）表格的基础。
    m = len(str1)
    n = len(str2)

    # 创建一个二维数组来存储编辑距离
    # 为什么是 (m+1) x (n+1)？ 动态规划表格的维度比字符串长度多1，这是为了包含空字符串的情况（即 i=0 或 j=0）。
    # dp[i][j] 的含义：它代表将 str1 的 前 i 个字符 转换为 str2 的 前 j 个字符 所需的最少操作数。
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化第一行和第一列
    # 这是DP算法最关键的一步——处理空字符串的情况：
    # 第一列 dp[i][0]：要将 str1 的前 i 个字符变成空字符串，唯一的办法是删除所有字符。因此，dp[i][0] = i。
    # 第一行 dp[j]：要将空字符串变成 str2 的前 j 个字符，唯一的办法是插入所有字符。因此，dp[j] = j。
    for i in range(m + 1):
        dp[i][0] = i  # 删除操作
    for j in range(n + 1):
        dp[0][j] = j  # 插入操作

    # 核心逻辑：状态转移方程
    # 这是整个算法的灵魂，它通过双重循环从左上到右下填充表格的每个单元格。每个单元格的值取决于它相邻的三个单元格。
    # 场景一：字符相同
    # 如果 str1[i-1] == str2[j-1]，意味着当前字符已经匹配，不需要任何操作。因此，当前子问题的最小编辑距离就等于两个字符串都去掉这个字符后的编辑距离，即 dp[i][j] = dp[i-1][j-1]。
    # 场景二：字符不同
    # 如果字符不同，我们需要从三种操作中选择一个代价最小的，然后加上1（表示执行了这次操作）：

    # 删除操作 (dp[i-1][j] + 1)：删除 str1 的第 i 个字符。这意味着 str1 的前 i-1 个字符已经能和 str2 的前 j 个字符匹配了，所以代价是 dp[i-1][j] 再加上一次删除操作。
    # 插入操作 (dp[i][j-1] + 1)：在 str1 中插入一个字符（等于 str2[j-1]）。这意味着 str1 的前 i 个字符已经和 str2 的前 j-1 个字符匹配了，插入一个字符就能完全匹配，所以代价是 dp[i][j-1] 再加上一次插入操作。
    # 替换操作 (dp[i-1][j-1] + 1)：将 str1 的第 i 个字符替换为 str2 的第 j 个字符。这意味着 str1 的前 i-1 个字符和 str2 的前 j-1 个字符已经匹配，只需替换最后一个字符，所以代价是 dp[i-1][j-1] 再加上一次替换操作。

    # 最后，min() 函数会从这三个候选值中选出最小的一个作为 dp[i][j]，这就是动态规划的“最优子结构”特性。
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 字符相同，不需要操作
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,      # 删除操作
                               dp[i][j - 1] + 1,      # 插入操作
                               dp[i - 1][j - 1] + 1)  # 替换操作

    # 最终，dp[m][n] 位于表格的右下角，它代表了将整个 str1（前 m 个字符）转换为整个 str2（前 n 个字符）所需的最少操作数，也就是我们要求的编辑距离。

    # 一个简单例子
    # 假设我们计算 "horse" 和 "ros" 的编辑距离。

    # 初始化后，第一行是 [0, 1, 2, 3]，第一列是 [0, 1, 2, 3, 4, 5]。
    # 经过动态规划填充，最终的DP表格会变成这样（实际计算值）：
    #     ''   r   o   s
    # ''  0   1   2   3
    # h   1   1   2   3
    # o   2   2   1   2
    # r   3   2   2   2
    # s   4   3   3   2
    # e   5   4   4   3

    # 最终结果是 dp[3] = 3，这与题目给出的最少三步操作（h->r替换，删除r，删除e）完全一致。

    # 总结
    # 表格行号作用1-2定义函数，获取字符串长度5创建 (m+1) x (n+1) 的二维DP表，所有值初始化为08-10初始化边界：第一列表示删除，第一行表示插入13-19核心循环：遍历每个 i 和 j，根据字符是否相同，应用状态转移方程16-18如果字符不同，从删除、插入、替换中选择代价最小的操作21-22返回最终计算出的编辑距离 dp[m][n]
    # 这段代码通过经典的动态规划思想，将复杂问题分解为一系列可重复计算的子问题，高效地解决了字符串相似度度量问题，是面试和实际应用（如拼写检查、DNA序列分析）中的高频考点。希望这段解读能帮助你彻底理解它！
    return dp[m][n]

# 引入三方库做效果对比
import Levenshtein
import timeit
import random
import string
def compare_edit_distance(str1, str2):
    # 计算代码执行耗时
    custom_time = timeit.timeit(
        lambda: edit_distance(str1, str2),
        number=10
    )
    lib_time = timeit.timeit(
        lambda: Levenshtein.distance(str1, str2),
        number=10
    )

    print(f"Custom Edit Distance : {custom_time:.6f}s")
    print(f"Levenshtein Distance : {lib_time:.6f}s")
    print(f"加速比 : {custom_time/lib_time:.2f}x")

# 测试代码
if __name__ == "__main__":
    str1 = ''.join(random.choices(string.ascii_lowercase, k=1000))
    str2 = ''.join(random.choices(string.ascii_lowercase, k=1000))
    compare_edit_distance(str1, str2)
