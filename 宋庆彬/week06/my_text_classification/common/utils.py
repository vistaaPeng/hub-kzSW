"""通用工具：设备选择 + 中文字体 + 计时器"""

import time
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def get_device() -> torch.device:
    """自动选择最优设备：MPS > CUDA > CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_chinese_font() -> str | None:
    """检测系统可用的中文字体，找不到返回 None。"""
    candidates = [
        "PingFang SC", "Heiti SC", "STHeiti", "SimHei",
        "Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def setup_plot_font():
    """配置 matplotlib 中文字体。"""
    cn_font = find_chinese_font()
    if cn_font:
        plt.rcParams["font.family"] = cn_font
    plt.rcParams["axes.unicode_minus"] = False


class Timer:
    """上下文计时器。"""

    def __init__(self, label: str = ""):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"[Timer] {self.label}: {elapsed:.1f}s" if self.label else f"{elapsed:.1f}s")
