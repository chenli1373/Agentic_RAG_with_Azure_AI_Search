from typing import List
import re
import time
import asyncio
from config.settings import ToolRuntime

# RAG-Fusion 核心算法： Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(results: list[list[dict]], k=60) -> List[dict]:
    """ 融合多个查询的结果，通过排名加权计算文档分数，并得到最终的 reranked 文档列表 
        score(d) = sum(1 / (k + rank(d, q_i))) for i in range(len(queries))
    """

    # 初始化分数字典
    fused_scores = {}
    fused_docs = {}

    # 遍历每个查询的结果
    for docs in results:
        # 遍历每个文档及其排名
        for rank, doc in enumerate(docs):
            doc_id = doc["content"]

            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                fused_docs[doc_id] = doc
            
            # 通过 RRF 公式更新分数
            fused_scores[doc_id] += 1 / (k + rank)
    
    # 将文档分数排序
    sorted_docs = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [fused_docs[doc_id] for doc_id, _ in sorted_docs]

def rerank(query: str, docs: list[dict], top_k: int = 5) -> List[dict]:
    # 将 query 分词并去重
    query_tokens = set(tokenize(query))

    # 去除重复 document
    seen = set()
    # 评分
    scored = []

    for doc in docs:
        content = doc["content"]

        h = hash(content)
        if h in seen:
            continue
        seen.add(h)

        # 文档分词
        doc_tokens = tokenize(content)

        score = 0
        for t in query_tokens:
            # query token 在 doc 中出现多少次
            tf = doc_tokens.count(t)
            # 关键词出现越多 + 文档越短 -> 分数越高
            if tf > 0:
                score += tf / (len(doc_tokens) + 1)
        scored.append((score, doc))
    
    scored.sort(key=lambda x: x[0], reverse=True)

    return [d for _, d in scored[: top_k]]

# 简单的分词器
def tokenize(text: str) -> List[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())

async def retrieval(rewritten_query: str, multi_queries: List[str], runtime: ToolRuntime):
    """
    search -> RRF -> Rerank
    """
    start = time.time()

    tasks = [
        runtime.resources.search_client.search(q)
        for q in multi_queries
    ]

    retrieval_docs = await asyncio.gather(*tasks)

    print("\n===== retrieval results =====")
    for i, r in enumerate(retrieval_docs):
        print(f"\n--- query {i} ---")
        print(r)
    print("=================================\n")

    RRF_docs = reciprocal_rank_fusion(retrieval_docs)

    print("\n===== RRF Fusion =====")
    for i, doc in enumerate(RRF_docs):
        print(f"\n[{i}]")
        print(doc)
    print("=========================\n")

    reranked_docs = rerank(rewritten_query, RRF_docs, top_k=5)

    print("\n===== Reranked Result =====")
    for i, doc in enumerate(reranked_docs):
        score_preview = doc.get("score", None) if isinstance(doc, dict) else None
        print(f"\n[{i}] score={score_preview}")
        print(doc)
    print("==============================\n")

    latency = int((time.time() - start) * 1000)
    print(f"\n===== Retrieval & Reranking Latency: {latency} ms =====\n")

    return {
        "retrieval_docs": retrieval_docs,
        "RRF_docs": RRF_docs,
        "reranked_docs": reranked_docs
    }