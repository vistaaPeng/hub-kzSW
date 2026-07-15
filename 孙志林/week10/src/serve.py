"""
RAG问答系统服务接口（Flask）

启动方式：
  python serve.py
  python serve.py --host 0.0.0.0 --port 8080

API接口：
  POST /api/query
    参数：{ "query": "贵州茅台2023年营收", "stock": "600519", "year": "2023" }
    返回：{ "answer": "...", "sources": [...], "retrieved": [...] }

  GET /api/health
    返回：{ "status": "ok" }
"""

import json
import logging
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify, render_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "src" / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), template_folder=str(STATIC_DIR))

rag_pipeline = None


def init_rag():
    """初始化RAG流水线"""
    global rag_pipeline
    from rag_pipeline import rag_pipeline as pipeline_func
    rag_pipeline = pipeline_func
    logger.info("RAG pipeline initialized")


@app.route("/")
def index():
    """前端页面"""
    return render_template("index.html")


@app.route("/api/health")
def health():
    """健康检查"""
    return jsonify({"status": "ok", "rag_ready": rag_pipeline is not None})


@app.route("/api/query", methods=["POST"])
def api_query():
    """问答API接口"""
    try:
        data = request.get_json()
        query = data.get("query", "")
        stock = data.get("stock", "")
        year = data.get("year", "")

        if not query:
            return jsonify({"error": "query is required"}), 400

        logger.info(f"Received query: {query} (stock={stock}, year={year})")

        result = rag_pipeline(
            query=query,
            stock_code=stock if stock else None,
            year=year if year else None,
        )

        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved": [
                {k: v for k, v in r.items() if k not in ["vec_score", "bm25_score", "rrf_score", "rerank_score"]}
                for r in result.get("retrieved", [])
            ],
        })

    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stocks")
def api_stocks():
    """获取支持的股票列表"""
    stocks = [
        {"code": "600519", "name": "贵州茅台"},
        {"code": "000858", "name": "五粮液"},
        {"code": "300750", "name": "宁德时代"},
        {"code": "002415", "name": "海康威视"},
        {"code": "601318", "name": "中国平安"},
    ]
    return jsonify(stocks)


@app.route("/api/years")
def api_years():
    """获取支持的年份列表"""
    return jsonify(["2021", "2022", "2023"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Service")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    init_rag()
    logger.info(f"Starting RAG service on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)