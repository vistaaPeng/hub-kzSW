"""
数据下载模块：从巨潮资讯网下载上市公司年报PDF

巨潮资讯网（cninfo.com.cn）是证监会指定的上市公司信息披露平台，数据公开合法。

使用方式：
  python download_reports.py
  python download_reports.py --stock 600519 --year 2023
"""

import os
import json
import logging
import argparse
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdf"
RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)

CNINFO_API = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
PDF_BASE_URL = "http://file.finance.sina.com.cn/211.154.219.97:9494/MRGG/CNSESH_STOCK/"

COMPANIES = [
    {"code": "600519", "name": "贵州茅台", "exchange": "SH"},
    {"code": "000858", "name": "五粮液", "exchange": "SZ"},
    {"code": "300750", "name": "宁德时代", "exchange": "SZ"},
    {"code": "002415", "name": "海康威视", "exchange": "SZ"},
    {"code": "601318", "name": "中国平安", "exchange": "SH"},
]

YEARS = ["2021", "2022", "2023"]


def download_report(stock_code: str, stock_name: str, year: str, exchange: str):
    """下载单个公司单年的年报PDF"""
    pdf_name = f"{stock_code}_{year}_{stock_name}_{year}年年度报告.pdf"
    pdf_path = RAW_PDF_DIR / pdf_name

    if pdf_path.exists():
        logger.info(f"Already downloaded: {pdf_name}")
        return str(pdf_path)

    try:
        params = {
            "pageNum": "1",
            "pageSize": "30",
            "tabName": "fulltext",
            "searchkey": f"{stock_name} {year}年年度报告",
            "secid": f"{exchange}{stock_code}",
            "category": "",
            "trade": "",
            "seDate": f"{year}-01-01~{year}-12-31",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }

        resp = requests.post(CNINFO_API, data=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("announcements", []):
            if "年度报告" in item.get("announcementTitle", "") and year in item.get("announcementTime", ""):
                pdf_url = item.get("adjunctUrl", "")
                if pdf_url:
                    logger.info(f"Downloading: {pdf_name}")
                    pdf_resp = requests.get(f"http://www.cninfo.com.cn/{pdf_url}", timeout=60)
                    pdf_resp.raise_for_status()

                    with open(pdf_path, "wb") as f:
                        f.write(pdf_resp.content)

                    logger.info(f"Saved to: {pdf_path}")
                    return str(pdf_path)

        logger.warning(f"No annual report found for {stock_name} {year}")
    except Exception as e:
        logger.error(f"Failed to download {stock_name} {year} report: {e}")

    return None


def download_all_reports():
    """下载所有公司所有年份的年报"""
    downloaded = []

    for company in COMPANIES:
        for year in YEARS:
            pdf_path = download_report(
                company["code"], company["name"], year, company["exchange"]
            )
            if pdf_path:
                downloaded.append({
                    "stock_code": company["code"],
                    "stock_name": company["name"],
                    "year": year,
                    "pdf_path": pdf_path,
                })

    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(downloaded, f, ensure_ascii=False, indent=2)

    logger.info(f"Download complete! {len(downloaded)} reports saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download annual reports")
    parser.add_argument("--stock", type=str, help="Stock code (e.g., 600519)")
    parser.add_argument("--year", type=str, help="Year (e.g., 2023)")
    args = parser.parse_args()

    if args.stock and args.year:
        for company in COMPANIES:
            if company["code"] == args.stock:
                download_report(company["code"], company["name"], args.year, company["exchange"])
                break
    else:
        download_all_reports()