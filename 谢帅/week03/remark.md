train_ni_position_cls.py
"你"字位置多分类任务 —— RNN / LSTM / GRU 对比

GRU（门控循环单元）：RNN 的改进变体，通过"重置门"和"更新门"两个门控机制控制信息的保留与遗忘。
- 重置门 r：决定遗忘多少过去的记忆，r 接近 0 时忽略旧状态，相当于"重新开始"
- 更新门 z：决定保留多少旧状态、接入多少新状态，z 接近 1 时直接传递旧状态
- 相比 LSTM 少一个门且无独立细胞状态，参数更少、训练更快，多数任务效果接近 LSTM

任务：5 字中文文本中包含"你"，"你"在第几位 → 第几类（1~5，即 5 分类）
模型：Embedding → RNN/LSTM/GRU → MaxPool → Linear → CrossEntropyLoss
优化：Adam (lr=1e-3)   CPU 即可运行

依赖：torch >= 2.0   (pip install torch)
