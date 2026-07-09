import os
import json
import logging
from flask import Flask, request, jsonify
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
pipeline = None


def init_pipeline():
    global pipeline
    from rag_pipeline import RAGPipeline
    pipeline = RAGPipeline(use_bm25=True)
    logger.info("RAG Pipeline 初始化完成")


@app.route("/api/query", methods=["POST"])
def query():
    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 500
    
    try:
        data = request.get_json()
        question = data.get("question", "")
        doc_type = data.get("doc_type", None)
        
        if not question:
            return jsonify({"error": "缺少 question 参数"}), 400
        
        filter_meta = {}
        if doc_type:
            filter_meta["doc_type"] = doc_type
        if not filter_meta:
            filter_meta = None
        
        result = pipeline.query(question, filter_meta=filter_meta, verbose=True)
        
        return jsonify({
            "question": question,
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "retrieved_count": len(result.get("retrieved", [])),
        })
    except Exception as e:
        logger.error(f"查询失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status": "running",
        "pipeline_initialized": pipeline is not None,
    })


if __name__ == "__main__":
    init_pipeline()
    app.run(host="0.0.0.0", port=5000, debug=False)