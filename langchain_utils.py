from langchain_core.documents import Document
from langchain_classic.load import dumps, loads

import os
from typing import List, Dict
import re
import json


async def Query_Rewrite(query: str, history: List[dict], llm_client):
    history = history or []

    if len(history) == 0:
        context = "No history yet."
    else:
        context = "\n".join(
            f"{h['role']}: {h['content']}" for h in history
        )

    prompt = f"""
    You are a smart query rewriting assistant. 
    Rewrite the user's query to make it clear, concise, and optimized for information retrieval or downstream processing based on the given context. 
    Keep the meaning the same, remove ambiguity, and make it more specific if possible. 
    Do not add extra information that wasn't implied in the original query.

    Original Query: "{query}"
    context: {context}
    Rewritten Query:
    """
    response = await llm_client.chat.completions.create(
        model=os.getenv("AI_MODEL_DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": "You are a smart query rewriting assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

async def generate_multi_queries(query: str, llm_client) -> List[str]:
    prompt = f"""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {query}
    """
    response = await llm_client.chat.completions.create(
        model=os.getenv("AI_MODEL_DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": "You are a helpful query generation assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    result = response.choices[0].message.content

    queries = [q.strip() for q in result.split("\n") if q.strip()]

    return queries

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

# ground truth
def build_relevant_docs_from_answer(answer: str) -> List[Dict]:
    """
    将 QA 的 answer 直接作为 relevant_docs 用于离线评估
    """
    return [
        {
            "content": answer,
            "metadata": {"source": "qa_ground_truth"}
        }
    ]

class QAIterator:
    def __init__(self, qa_path: str):
        with open(qa_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.dataset = data["dataset"]
        self.index = 0

    def has_next(self):
        return self.index < len(self.dataset)
    
    def next(self):
        item = self.dataset[self.index]
        self.index += 1
        return item
