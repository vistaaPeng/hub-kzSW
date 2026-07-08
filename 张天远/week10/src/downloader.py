"""
数据下载器 —— 从 mdBook 文档网站递归下载章节页面。

mdBook 生成的 HTML 侧边栏结构：
<ol class="chapter">
  <li><a href="ch01.html">第一章</a></li>
  <li><a href="ch02.html">第二章</a>
    <ol>
      <li><a href="ch02-01.html">2.1 节</a></li>
    </ol>
  </li>
</ol>
"""

import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
}
REQUEST_DELAY = 0.5  # 礼貌间隔（秒）


def extract_chapter_links(html: str, base_url: str) -> list[str]:
    """
    从 mdBook HTML 侧边栏提取所有章节链接。

    Args:
        html: 页面 HTML 源码
        base_url: 用于解析相对路径的基础 URL

    Returns:
        去重后的绝对 URL 列表，仅包含 .html 结尾的链接
    """
    soup = BeautifulSoup(html, "html.parser")
    chapter_ol = soup.find("ol", class_="chapter")

    if chapter_ol is None:
        return []

    links = set()
    for a_tag in chapter_ol.find_all("a", href=True):
        href = a_tag["href"].strip()
        # 跳过锚点、非 HTML 链接
        if href.startswith("#"):
            continue
        if not href.endswith(".html"):
            continue
        full_url = urljoin(base_url, href)
        # 确保是同域链接
        if urlparse(full_url).netloc == urlparse(base_url).netloc:
            links.add(full_url)

    return sorted(links)


def download_page(url: str, output_dir: Path, delay: float = REQUEST_DELAY) -> Path:
    """
    下载单个 HTML 页面到本地。

    Args:
        url: 页面 URL
        output_dir: 输出目录（Path 对象）
        delay: 请求间隔（秒）

    Returns:
        保存的文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 用 URL 路径的最后一部分做文件名
    parsed = urlparse(url)
    filename = Path(parsed.path).name
    if not filename.endswith(".html"):
        filename = "index.html"

    filepath = output_dir / filename

    time.sleep(delay)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding  # 自动检测编码

    filepath.write_text(resp.text, encoding="utf-8")
    return filepath


def download_book(root_url: str, output_dir: str) -> list[Path]:
    """
    下载整本 mdBook 的所有章节页面。

    Args:
        root_url: mdBook 的根页面 URL（如 https://rustwiki.org/zh-CN/book/）
        output_dir: 本地输出目录

    Returns:
        已下载的文件路径列表
    """
    out_dir = Path(output_dir)
    downloaded = []

    print(f"📥 下载: {root_url}")

    # 第一步：下载根页面，提取侧边栏链接
    root_path = download_page(root_url, out_dir, delay=0)
    downloaded.append(root_path)
    root_html = root_path.read_text(encoding="utf-8")

    chapter_urls = extract_chapter_links(root_html, root_url)
    print(f"  发现 {len(chapter_urls)} 个章节链接")

    # 第二步：逐个下载章节页面
    for i, url in enumerate(chapter_urls, 1):
        try:
            path = download_page(url, out_dir)
            downloaded.append(path)
            print(f"  [{i}/{len(chapter_urls)}] ✓ {Path(url).name}")
        except Exception as e:
            print(f"  [{i}/{len(chapter_urls)}] ✗ {Path(url).name}: {e}")

    return downloaded
