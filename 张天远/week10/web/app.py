#!/usr/bin/env python
"""
StreamLit 前端 —— RAG 查询界面
端口: 8501
后端 API: http://localhost:8000

页面:
  - 查询页: 输入框 + 回答 + 来源表格
  - 评估页: 四指标卡片 + 每题详情
  - 设置页: top_k / rerank / rewrite
  - 历史页: 查询日志
"""

import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── 页面配置 (暗色主题) ──────────────────────────────
st.set_page_config(
    page_title="RAG Scratch Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 缓存：避免每次交互都重新读文件/调 API ─────────────
@st.cache_data(ttl=30)
def cached_load_questions():
    q_path = Path("evaluation/questions_60.json")
    if q_path.exists():
        return json.loads(q_path.read_text(encoding="utf-8"))
    return None

@st.cache_data(ttl=10)
def cached_load_history():
    h_path = Path("logs/queries.jsonl")
    entries = []
    if h_path.exists():
        for line in h_path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries

@st.cache_data(ttl=60)
def cached_api_health():
    try:
        return requests.get(f"{API_BASE}/health", timeout=3).json()
    except Exception:
        return None

# ── 暗色主题 CSS ─────────────────────────────────────
st.markdown("""
<style>
    /* 整体暗色 */
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }
    /* 按钮 */
    .stButton > button {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #30363d;
        border-color: #8b949e;
        color: #f0f6fc;
    }
    .stButton > button:active {
        background-color: #1f6feb;
        border-color: #1f6feb;
    }
    /* 文本输入框 */
    .stTextInput > div > div > input {
        background-color: #0d1117;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #58a6ff;
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.3);
    }
    /* 数据表格 */
    .stDataFrame {
        background-color: #161b22;
    }
    .stDataFrame th {
        background-color: #21262d !important;
        color: #c9d1d9 !important;
    }
    .stDataFrame td {
        background-color: #0d1117 !important;
        color: #c9d1d9 !important;
    }
    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px 20px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover {
        border-color: #58a6ff;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #58a6ff, #3fb950);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 4px;
    }
    /* 来源卡片 */
    .source-item {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .source-item:hover {
        border-color: #58a6ff;
    }
    /* 回答框 */
    .answer-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 16px 0;
        line-height: 1.7;
    }
    /* 标题 */
    h1, h2, h3, h4 {
        color: #f0f6fc !important;
    }
    /* 展开器 */
    .streamlit-expanderHeader {
        background-color: #21262d !important;
        border-radius: 8px !important;
    }
    /* 滚动条 */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0d1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }
    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 20px;
        color: #484f58;
        font-size: 0.8rem;
        border-top: 1px solid #21262d;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# ── 后端 API 基础 URL ────────────────────────────────
API_BASE = "http://localhost:8000"

# ── Session State 初始化 ─────────────────────────────
if "history" not in st.session_state:
    # 从磁盘加载持久化历史
    history_path = Path("logs/queries.jsonl")
    st.session_state.history = cached_load_history()
if "settings" not in st.session_state:
    st.session_state.settings = {
        "top_k": 10,
        "rerank": False,
        "rewrite": True,
    }
if "current_page" not in st.session_state:
    st.session_state.current_page = "查询"
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "eval_running" not in st.session_state:
    st.session_state.eval_running = False


# ── API 辅助函数 ─────────────────────────────────────
def api_post(endpoint: str, data: dict, timeout: int = 30) -> Optional[dict]:
    """POST 请求封装"""
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(f"❌ 无法连接到后端 API ({API_BASE})。请确保 FastAPI 服务已启动。")
    except requests.Timeout:
        st.error(f"⏱️ 请求超时 ({timeout}s)。后端可能负载过高。")
    except requests.HTTPError as e:
        st.error(f"🌐 HTTP 错误: {e}")
    except Exception as e:
        st.error(f"⚠️ 请求异常: {e}")
    return None


def api_get(endpoint: str, timeout: int = 10) -> Optional[dict]:
    """GET 请求封装"""
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.warning(f"⚠️ 无法连接到后端 API ({API_BASE})。")
    except Exception as e:
        st.warning(f"⚠️ 请求异常: {e}")
    return None


def add_to_history(question: str, answer: str, sources: list, rewrites: list,
                   top_k: int, rerank: bool, rewrite: bool):
    """记录查询到历史"""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "sources": sources,
        "rewrites": rewrites,
        "top_k": top_k,
        "rerank": rerank,
        "rewrite": rewrite,
    }
    st.session_state.history.insert(0, entry)
    # 保留最近 100 条
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[:100]
    # 持久化到磁盘
    log_path = Path("logs/queries.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Sidebar 导航 ─────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0;">
        <h1 style="font-size:1.6rem; margin:0;">🔍 RAG Dashboard</h1>
        <p style="color:#8b949e; font-size:0.8rem; margin:4px 0;">Rust 文档智能问答</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    pages = {
        "查询": "📝",
        "评估": "📊",
        "设置": "⚙️",
        "历史": "📋",
    }

    for page_name, icon in pages.items():
        is_active = st.session_state.current_page == page_name
        btn_style = (
            "background-color: #1f6feb; border-color: #1f6feb; color: #fff !important;"
            if is_active else ""
        )
        if st.button(
            f"{icon}  {page_name}",
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.current_page = page_name
            st.rerun()

    st.markdown("---")

    # 健康检查
    health = cached_api_health()
    if health:
        st.success(f"✅ API 在线 | 文档块: {health.get('chunks', 'N/A')}")
    else:
        st.error("❌ API 离线")

    st.markdown("---")
    st.markdown(
        '<p style="color:#484f58; font-size:0.7rem; text-align:center;">'
        'StreamLit v1.x | FastAPI :8000</p>',
        unsafe_allow_html=True,
    )

# ── 页面路由 ─────────────────────────────────────────
page = st.session_state.current_page

# ═══════════════════════════════════════════════════════
# 📝 查询页
# ═══════════════════════════════════════════════════════
if page == "查询":
    st.title("📝 RAG 查询")
    st.markdown("输入 Rust 相关问题，获取基于文档检索的 AI 回答。")

    # 查询表单（Enter 触发提交）
    with st.form("query_form"):
        col1, col2 = st.columns([5, 1])
        with col1:
            question = st.text_input(
                "提问",
                placeholder="例如：Rust 中的所有权规则是什么？",
                label_visibility="collapsed",
                key="query_input"
            )
        with col2:
            submit = st.form_submit_button("🚀 查询", use_container_width=True)

    # 快捷设置行
    with st.expander("⚡ 快捷设置", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            quick_top_k = st.slider("Top-K", 1, 20, st.session_state.settings["top_k"], key="quick_top_k")
        with c2:
            quick_rerank = st.toggle("ReRank 重排", st.session_state.settings["rerank"], key="quick_rerank")
        with c3:
            quick_rewrite = st.toggle("Query Rewrite 改写", st.session_state.settings["rewrite"], key="quick_rewrite")
        if st.button("💾 保存到全局设置", key="save_quick"):
            st.session_state.settings.update({
                "top_k": quick_top_k,
                "rerank": quick_rerank,
                "rewrite": quick_rewrite,
            })
            st.success("设置已保存！")

    # 执行查询
    if submit and question.strip():
        settings = st.session_state.settings
        with st.spinner("🤔 正在检索并生成回答..."):
            payload = {
                "question": question.strip(),
                "rerank": settings["rerank"],
            }
            result = api_post("/query", payload, timeout=60)

        if result:
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            rewrites = result.get("rewrites", [])

            # 记录到历史
            add_to_history(
                question=question.strip(),
                answer=answer,
                sources=sources,
                rewrites=rewrites,
                top_k=settings["top_k"],
                rerank=settings["rerank"],
                rewrite=settings["rewrite"],
            )

            # 显示回答
            st.markdown("### 💡 回答")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            # 改写信息
            if rewrites:
                with st.expander(f"🔄 查询改写 ({len(rewrites)} 个变体)", expanded=False):
                    for i, rw in enumerate(rewrites, 1):
                        st.markdown(f"- **变体 {i}**: `{rw}`")

            # 来源表格
            st.markdown("### 📚 参考来源")
            if sources:
                # 表格 + 卡片双视图
                tab1, tab2 = st.tabs(["📋 表格视图", "🃏 卡片视图"])

                with tab1:
                    df = pd.DataFrame(sources)
                    # 选择展示的列
                    display_cols = ["rank", "score", "source_url", "headings", "preview"]
                    available_cols = [c for c in display_cols if c in df.columns]
                    st.dataframe(
                        df[available_cols],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "rank": st.column_config.NumberColumn("排名", width="small"),
                            "score": st.column_config.NumberColumn("相关度", format="%.4f"),
                            "source_url": st.column_config.LinkColumn("来源", width="medium"),
                            "headings": st.column_config.TextColumn("标题", width="medium"),
                            "preview": st.column_config.TextColumn("预览", width="large"),
                        },
                    )

                with tab2:
                    for src in sources:
                        with st.container():
                            st.markdown(f"""
                            <div class="source-item">
                                <strong>#{src.get('rank', '?')}</strong>
                                &nbsp;得分: <code>{src.get('score', 0):.4f}</code>
                                &nbsp;|&nbsp; 📄 <em>{src.get('headings', '无标题')}</em>
                                <br>
                                <a href="{src.get('source_url', '')}" target="_blank" style="color:#58a6ff;">🔗 查看原文</a>
                                <p style="margin-top:8px; font-size:0.9rem;">{src.get('preview', '')}...</p>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("未找到相关来源。")

    elif submit and not question.strip():
        st.warning("请输入问题再查询。")

# ═══════════════════════════════════════════════════════
# 📊 评估页
# ═══════════════════════════════════════════════════════
elif page == "评估":
    st.title("📊 RAG 评估")
    st.markdown("评估 RAG 系统的四大核心指标：**Context Precision**、**MRR**、**Faithfulness**、**Answer Relevancy**。")

    # 评估运行中 → 所有控件禁用，防止误触中断
    eval_is_running = st.session_state.get("eval_running", False)
    if eval_is_running:
        st.warning("⏳ 评估正在运行，页面已锁定...")

    results_dir = Path("evaluation/results")
    if results_dir.exists():
        history_files = sorted(results_dir.glob("eval_*.json"), reverse=True)
        if history_files:
            hist_options = ["最新结果"] + [f.name.replace("eval_", "").replace(".json", "") for f in history_files]
            selected_hist = st.selectbox("📜 历史评估", hist_options, key="eval_history")
            if selected_hist != "最新结果":
                hist_path = results_dir / f"eval_{selected_hist}.json"
                if hist_path.exists() and st.session_state.get("eval_loaded_path") != str(hist_path):
                    loaded = json.loads(hist_path.read_text(encoding="utf-8"))
                    st.session_state.eval_results = loaded.get("details", [])
                    st.session_state.eval_loaded_path = str(hist_path)
                    st.rerun()

    # ── 评估控制区 ──
    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        st.markdown("""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:8px 0;">
            <strong>📋 评估流程</strong>：选择或输入测试问题集 → 对每个问题执行 RAG 查询 →
            计算四项指标 → 汇总报告。需要 DeepSeek API 支持 Faithfulness 和 Answer Relevancy 指标。
        </div>
        """, unsafe_allow_html=True)
    with col_ctrl2:
        run_eval = st.button("🚀 运行评估", use_container_width=True, type="primary", key="run_eval_btn")
        if st.button("🗑️ 清除结果", use_container_width=True, key="clear_eval"):
            st.session_state.eval_results = None
            st.rerun()

    # ── 测试问题集 ──
    with st.expander("📝 测试问题集", expanded=False):
        import json as _json
        q_path = Path("evaluation/questions_60.json")
        if q_path.exists():
            raw = cached_load_questions() or json.loads(q_path.read_text(encoding="utf-8"))
            questions = [
                {"id": q.get("question_id", f"Q{i+1}"),
                 "question": q["question"],
                 "keywords": q.get("keywords", [])}
                for i, q in enumerate(raw)
            ]
        else:
            questions = [
                {"id": "Q1", "question": "Rust 中的所有权规则是什么？", "keywords": ["所有权", "ownership"]},
                {"id": "Q2", "question": "什么是生命周期标注？", "keywords": ["生命周期", "lifetime"]},
                {"id": "Q3", "question": "Rust 的 trait 和接口有什么区别？", "keywords": ["trait", "接口"]},
                {"id": "Q4", "question": "如何用 Rust 处理错误？", "keywords": ["错误处理", "Result", "panic"]},
                {"id": "Q5", "question": "Rust 中的智能指针有哪些？", "keywords": ["智能指针", "Box", "Rc"]},
            ]
        st.caption(f"共 {len(questions)} 道题")

        # 上传新问题集替换
        uploaded = st.file_uploader("📤 上传新问题集 (JSON)", type=["json"], key="eval_upload")
        if uploaded:
            try:
                raw = json.loads(uploaded.read().decode("utf-8"))
                questions = [
                    {"id": q.get("question_id", f"Q{i+1}"),
                     "question": q["question"],
                     "keywords": q.get("keywords", [])}
                    for i, q in enumerate(raw)
                ]
                st.session_state.eval_questions = questions
                st.success(f"已替换为 {len(questions)} 道题")
            except Exception:
                st.error("JSON 解析失败，保留原问题集")

        # 翻页浏览
        page_size = 10
        total_pages = max(1, (len(questions) + page_size - 1) // page_size)
        if "eval_page" not in st.session_state:
            st.session_state.eval_page = 0
        
        col_p1, col_p2, col_p3, col_p4 = st.columns([1, 1, 2, 1])
        with col_p1:
            if st.button("⬅ 上一页", disabled=(st.session_state.eval_page == 0), key="eval_prev"):
                st.session_state.eval_page = max(0, st.session_state.eval_page - 1)
                st.rerun()
        with col_p2:
            if st.button("下一页 ➡", disabled=(st.session_state.eval_page >= total_pages - 1), key="eval_next"):
                st.session_state.eval_page = min(total_pages - 1, st.session_state.eval_page + 1)
                st.rerun()
        with col_p3:
            st.caption(f"第 {st.session_state.eval_page + 1} / {total_pages} 页")
        with col_p4:
            new_page = st.number_input("跳转", 1, total_pages, st.session_state.eval_page + 1, key="eval_goto")
            if new_page != st.session_state.eval_page + 1:
                st.session_state.eval_page = new_page - 1
                st.rerun()

        start = st.session_state.eval_page * page_size
        st.json(questions[start:start + page_size])

    # ── 执行评估 ──
    if run_eval or st.session_state.get("eval_running"):
        if run_eval:
            st.session_state.eval_running = True
            st.session_state.eval_results = []  # 首次运行，初始化

        eval_qs = st.session_state.get("eval_questions", questions)
        results = st.session_state.eval_results  # 使用持久列表，重入不丢
        total = len(eval_qs)
        already_done = len(results)  # 已完成的题数

        progress_bar = st.progress(0)
        status_text = st.empty()
        live_table = st.empty()

        for i in range(already_done, total):
            q = eval_qs[i]
            status_text.text(f"正在评估: {q['question']} ({i+1}/{total})")
            progress_bar.progress((i + 1) / total)

            # 调用后端查询
            resp = api_post("/query", {"question": q["question"], "rerank": False}, timeout=60)
            if resp:
                sources = resp.get("sources", [])
                # 计算 Context Precision (关键词命中率)
                keywords_lower = [kw.lower() for kw in q.get("keywords", [])]
                hits = 0
                for src in sources:
                    preview = (src.get("preview", "") + " " + src.get("headings", "")).lower()
                    if any(kw in preview for kw in keywords_lower):
                        hits += 1
                cp = hits / len(sources) if sources else 0

                # 计算 MRR (简化：取第一个 source 的 rank 倒数)
                mrr = 1.0 / sources[0]["rank"] if sources else 0

                results.append({
                    "question_id": q["id"],
                    "question": q["question"],
                    "keywords": q.get("keywords", []),
                    "answer": resp.get("answer", ""),
                    "context_precision": round(cp, 4),
                    "mrr": round(mrr, 4),
                    "num_sources": len(sources),
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "faithfulness_detail": None,
                    "relevancy_detail": None,
                    "retrieved_chunks": [
                        {"rank": s.get("rank", i+1),
                         "score": s.get("score", 0),
                         "source_url": s.get("source_url", ""),
                         "headings": s.get("headings", ""),
                         "preview": s.get("preview", "")}
                        for i, s in enumerate(sources)
                    ],
                })
                # 调用后端计算 LLM 指标
                try:
                    safe_answer = resp.get("answer", "")[:1000].replace("\x00", "")
                    eval_resp = requests.post(
                        f"{API_BASE}/evaluate",
                        json={"question": q["question"][:200],
                              "answer": safe_answer,
                              "sources": [{"text": s.get("text", s.get("preview", "")), "preview": s.get("preview", "")[:300]}
                                         for s in sources[:8]]},
                        timeout=120,
                    )
                    if eval_resp.status_code == 200:
                        ed = eval_resp.json()
                        results[-1]["faithfulness"] = ed.get("faithfulness")
                        results[-1]["answer_relevancy"] = ed.get("answer_relevancy")
                        results[-1]["faithfulness_detail"] = ed.get("faithfulness_detail")
                        results[-1]["relevancy_detail"] = ed.get("relevancy_detail")
                except Exception:
                    pass
            else:
                results.append({
                    "question_id": q["id"],
                    "question": q["question"],
                    "answer": "API 调用失败",
                    "context_precision": 0,
                    "mrr": 0,
                    "num_sources": 0,
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "sources_preview": [],
                })

            # 实时更新结果表
            live_table.dataframe(
                pd.DataFrame([{
                    "ID": r["question_id"],
                    "问题": r["question"][:30],
                    "Precision": r["context_precision"],
                    "MRR": r["mrr"],
                    "Faith.": f'{r.get("faithfulness"):.2f}' if r.get("faithfulness") is not None else "…",
                    "Relev.": f'{r.get("answer_relevancy"):.2f}' if r.get("answer_relevancy") is not None else "…",
                    "回答": (r.get("answer", "") or "")[:60],
                } for r in results]),
                use_container_width=True,
                hide_index=True,
            )

        # 持久化到磁盘
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path(f"evaluation/results/eval_{ts}.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "meta": {
                "timestamp": ts,
                "question_count": len(results),
                "index_fingerprint": {
                    "vectorstore_dir": str(Path("vectorstore")),
                    "vectorstore_files": [f.name for f in Path("vectorstore").glob("*") if f.is_file()],
                },
            },
            "summary": {
                "context_precision": sum(r["context_precision"] for r in results) / len(results) if results else 0,
                "mrr": sum(r["mrr"] for r in results) / len(results) if results else 0,
                "faithfulness": sum(r["faithfulness"] for r in results if r.get("faithfulness") is not None) / max(1, sum(1 for r in results if r.get("faithfulness") is not None)) if results else None,
                "answer_relevancy": sum(r["answer_relevancy"] for r in results if r.get("answer_relevancy") is not None) / max(1, sum(1 for r in results if r.get("answer_relevancy") is not None)) if results else None,
            },
            "details": results,
        }
        save_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        st.session_state.eval_results = results
        st.session_state.eval_report_path = str(save_path)
        st.session_state.eval_running = False
        progress_bar.empty()
        status_text.empty()
        st.rerun()

    # ── 评估结果展示 ──
    if st.session_state.eval_results:
        results = st.session_state.eval_results

        # 四指标卡片
        st.markdown("### 📈 指标总览")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        # 只计算有值的指标
        cp_vals = [r["context_precision"] for r in results]
        mrr_vals = [r["mrr"] for r in results]
        ff_vals = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
        ar_vals = [r["answer_relevancy"] for r in results if r["answer_relevancy"] is not None]

        avg_cp = sum(cp_vals) / len(cp_vals) if cp_vals else 0
        avg_mrr = sum(mrr_vals) / len(mrr_vals) if mrr_vals else 0
        avg_ff = sum(ff_vals) / len(ff_vals) if ff_vals else 0
        avg_ar = sum(ar_vals) / len(ar_vals) if ar_vals else 0

        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_cp:.2%}</div>
                <div class="metric-label">🎯 Context Precision<br><small>关键词命中率</small></div>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_mrr:.4f}</div>
                <div class="metric-label">📊 MRR<br><small>平均倒数排名</small></div>
            </div>
            """, unsafe_allow_html=True)

        with col_m3:
            faith_display = f"{avg_ff:.2%}" if ff_vals else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{faith_display}</div>
                <div class="metric-label">🔬 Faithfulness<br><small>忠实度（需 LLM）</small></div>
            </div>
            """, unsafe_allow_html=True)

        with col_m4:
            ar_display = f"{avg_ar:.2%}" if ar_vals else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{ar_display}</div>
                <div class="metric-label">🎯 Answer Relevancy<br><small>答案相关性（需 LLM）</small></div>
            </div>
            """, unsafe_allow_html=True)

        # ── 每题得分表（可排序定位薄弱项） ──
        st.markdown("### 📊 题目得分明细")
        df_scores = pd.DataFrame([{
            "ID": r["question_id"],
            "问题": r["question"][:50],
            "Precision": r["context_precision"],
            "MRR": r["mrr"],
            "Faith.": round(r["faithfulness"], 3) if r.get("faithfulness") is not None else None,
            "Relev.": round(r["answer_relevancy"], 3) if r.get("answer_relevancy") is not None else None,
        } for r in results])
        st.dataframe(
            df_scores,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Precision": st.column_config.NumberColumn(format="%.3f"),
                "MRR": st.column_config.NumberColumn(format="%.4f"),
                "Faith.": st.column_config.NumberColumn(format="%.3f"),
                "Relev.": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption("💡 点击列标题排序，快速定位最低分题目。")

        # 每题详情
        st.markdown("### 📋 每题详情")
        for r in results:
            with st.expander(f"**{r['question_id']}**: {r['question']}", expanded=False):
                det_col1, det_col2, det_col3, det_col4 = st.columns(4)
                with det_col1:
                    st.metric("Context Precision", f"{r['context_precision']:.2%}")
                with det_col2:
                    st.metric("MRR", f"{r['mrr']:.4f}")
                with det_col3:
                    st.metric("Faithfulness", f"{r['faithfulness']:.2%}" if r['faithfulness'] is not None else "N/A")
                with det_col4:
                    st.metric("Answer Relevancy", f"{r['answer_relevancy']:.2%}" if r['answer_relevancy'] is not None else "N/A")

                st.markdown(f"**来源数**: {r['num_sources']}")
                st.markdown(f"**回答摘要**: {r['answer'][:300]}...")

                if r.get("retrieved_chunks"):
                    st.markdown("**检索片段**:")
                    for rc in r["retrieved_chunks"][:8]:
                        st.caption(f"#{rc['rank']} [{rc.get('headings','')[:40]}] {rc['preview'][:120]}...")
                
                if r.get("faithfulness_detail") and r["faithfulness_detail"].get("claims"):
                    with st.expander("🔬 Faithfulness 声明核查", expanded=False):
                        for claim in r["faithfulness_detail"]["claims"]:
                            icon = "✅" if claim.get("supported") else "❌"
                            st.markdown(f"{icon} {claim.get('statement','')[:120]}")
                
                if r.get("relevancy_detail") and r.get("relevancy_detail", {}).get("generated_questions"):
                    with st.expander("🎯 Answer Relevancy 逆向问题", expanded=False):
                        for gq in r["relevancy_detail"]["generated_questions"]:
                            st.markdown(f"- {gq}")

    elif not st.session_state.eval_running:
        st.info("👆 点击「运行评估」开始评估，或展开「测试问题集」查看/修改问题。")

# ═══════════════════════════════════════════════════════
# ⚙️ 设置页
# ═══════════════════════════════════════════════════════
elif page == "设置":
    st.title("⚙️ 系统设置")
    st.markdown("配置 RAG 查询参数，影响检索和生成行为。")

    settings = st.session_state.settings

    # ── 检索设置 ──
    st.markdown("### 🔍 检索设置")
    st.markdown("---")

    col_set1, col_set2 = st.columns(2)

    with col_set1:
        new_top_k = st.slider(
            "Top-K 检索数量",
            min_value=1,
            max_value=20,
            value=settings["top_k"],
            step=1,
            help="每次检索返回的最相关文档块数量。值越大覆盖面越广但可能引入噪声。",
        )
        st.caption(f"当前: 返回前 **{new_top_k}** 个最相关结果")

    with col_set2:
        new_rerank = st.toggle(
            "ReRank 重排序",
            value=settings["rerank"],
            help="启用后使用重排序模型对检索结果进行二次排序，提高 Top-5 精度。会增加约 1-3 秒延迟。",
        )
        if new_rerank:
            st.success("✅ 已启用 — 检索后将进行重排序")
        else:
            st.info("ℹ️ 已关闭 — 使用原始检索排序")

    new_rewrite = st.toggle(
        "Query Rewrite 查询改写",
        value=settings["rewrite"],
        help="启用后使用 LLM 对原始问题进行多角度改写，提高检索召回率。",
    )
    if new_rewrite:
        st.success("✅ 已启用 — 将生成 3 个查询变体")
    else:
        st.info("ℹ️ 已关闭 — 仅使用原始查询")

    # ── 保存按钮 ──
    st.markdown("---")
    col_save1, col_save2, col_save3 = st.columns([1, 1, 2])
    with col_save1:
        if st.button("💾 保存设置", use_container_width=True, type="primary", key="save_settings"):
            st.session_state.settings.update({
                "top_k": new_top_k,
                "rerank": new_rerank,
                "rewrite": new_rewrite,
            })
            st.success("✅ 设置已保存！")
            st.rerun()
    with col_save2:
        if st.button("🔄 恢复默认", use_container_width=True, key="reset_settings"):
            st.session_state.settings = {
                "top_k": 10,
                "rerank": False,
                "rewrite": True,
            }
            st.success("✅ 已恢复默认设置！")
            st.rerun()

    # ── 当前配置预览 ──
    st.markdown("---")
    st.markdown("### 📋 当前生效配置")
    st.json(st.session_state.settings)

    # ── API 连接测试 ──
    st.markdown("---")
    st.markdown("### 🔗 API 连接")
    if st.button("🩺 测试连接", key="test_conn"):
        health = cached_api_health()
        if health:
            st.success(f"✅ 连接正常 | 状态: {health.get('status', 'ok')} | 文档块数: {health.get('chunks', 'N/A')}")
            st.json(health)
        else:
            st.error(f"❌ 无法连接到 {API_BASE}")

# ═══════════════════════════════════════════════════════
# 📋 历史页
# ═══════════════════════════════════════════════════════
elif page == "历史":
    st.title("📋 查询历史")
    st.markdown("查看本次会话中的所有查询记录。")

    col_h1, col_h2 = st.columns([3, 1])
    with col_h2:
        if st.button("🗑️ 清空历史", use_container_width=True, key="clear_history"):
            st.session_state.history = []
            st.success("历史已清空！")
            st.rerun()

    history = st.session_state.history

    if not history:
        st.info("📭 暂无查询记录。去「查询」页开始提问吧！")
    else:
        # 统计
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("总查询数", len(history))
        with col_stat2:
            avg_sources = (
                sum(len(h.get("sources", [])) for h in history) / len(history)
            )
            st.metric("平均来源数", f"{avg_sources:.1f}")
        with col_stat3:
            rerank_count = sum(1 for h in history if h.get("rerank"))
            st.metric("ReRank 使用", f"{rerank_count}/{len(history)}")

        st.markdown("---")

        # 历史列表
        for i, entry in enumerate(history):
            with st.expander(
                f"**#{len(history) - i}** | {entry['timestamp']} | {entry['question'][:60]}{'...' if len(entry['question']) > 60 else ''}",
                expanded=(i == 0),  # 最新一条默认展开
            ):
                col_hd1, col_hd2 = st.columns([3, 1])
                with col_hd1:
                    st.markdown(f"**❓ 问题**: {entry['question']}")
                with col_hd2:
                    st.caption(f"🕐 {entry['timestamp']}")

                st.markdown(f"**💡 回答**:")
                st.markdown(f'<div class="answer-box">{entry["answer"]}</div>', unsafe_allow_html=True)

                # 查询参数
                params = f"Top-K={entry.get('top_k', '?')} | ReRank={'✅' if entry.get('rerank') else '❌'} | Rewrite={'✅' if entry.get('rewrite') else '❌'}"
                st.caption(f"⚙️ 参数: {params}")

                if entry.get("rewrites"):
                    st.caption(f"🔄 改写: {', '.join(entry['rewrites'])}")

                # 来源
                sources = entry.get("sources", [])
                if sources:
                    st.markdown(f"**📚 来源 ({len(sources)})**:")
                    df_src = pd.DataFrame(sources)
                    display_cols = ["rank", "score", "headings", "preview"]
                    available_cols = [c for c in display_cols if c in df_src.columns]
                    st.dataframe(
                        df_src[available_cols],
                        use_container_width=True,
                        hide_index=True,
                    )

                # 重新查询按钮
                if st.button("🔄 重新查询", key=f"re_query_{i}"):
                    st.session_state["_requery"] = entry["question"]
                    st.session_state.current_page = "查询"
                    st.rerun()

    # ── 导出功能 ──
    if history:
        st.markdown("---")
        with st.expander("📥 导出历史", expanded=False):
            # 可选的导出列
            all_fields = {
                "timestamp": "时间",
                "question": "问题",
                "answer": "回答",
                "num_sources": "来源数",
                "top_k": "Top-K",
                "rerank": "重排开关",
                "rewrite": "改写开关",
                "rewrites": "改写内容",
                "sources": "完整来源",
            }
            st.caption("选择要导出的列")
            cols = st.columns(4)
            selected = {}
            for idx, (key, label) in enumerate(all_fields.items()):
                with cols[idx % 4]:
                    selected[key] = st.checkbox(label, value=key in ["timestamp", "question", "answer"], key=f"export_{key}")

            export_format = st.radio("格式", ["JSON", "CSV"], horizontal=True, key="export_fmt")
            if export_format == "JSON":
                # 按选中列过滤
                filtered = [{k: h.get(k, "") for k in selected if selected[k]} for h in history]
                export_data = json.dumps(filtered, ensure_ascii=False, indent=2)
                st.download_button(
                    "📥 下载 JSON", export_data,
                    file_name=f"rag_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )
            else:
                csv_rows = []
                for h in history:
                    row = {}
                    for k in selected:
                        if not selected[k]:
                            continue
                        if k == "num_sources":
                            row[k] = len(h.get("sources", []))
                        elif k == "sources":
                            row[k] = json.dumps(h.get("sources", []), ensure_ascii=False)
                        elif k == "rewrites":
                            row[k] = json.dumps(h.get("rewrites", []), ensure_ascii=False)
                        else:
                            val = h.get(k, "")
                            row[k] = str(val)[:1000] if k == "answer" else val
                    csv_rows.append(row)
                df_export = pd.DataFrame(csv_rows)
                st.download_button(
                    "📥 下载 CSV", df_export.to_csv(index=False).encode("utf-8"),
                    file_name=f"rag_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

# ── Footer ────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="custom-footer">'
    f'RAG Scratch Dashboard · StreamLit v1.x · FastAPI :8000 · {datetime.now().strftime("%Y-%m-%d %H:%M")}'
    '</div>',
    unsafe_allow_html=True,
)
