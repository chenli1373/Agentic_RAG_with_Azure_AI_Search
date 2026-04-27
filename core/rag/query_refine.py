from typing import List
import os
from config.settings import ToolRuntime

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

async def query_refine(query: str, runtime: ToolRuntime):
    """
    query rewrite -> multi-query
    """
    history = runtime.session.history or []

    print("\n===== Conversation History =====")
    if not history:
        print("There is no history yet.")
    else:
        for msg in history:
            print(f"{msg['role']}: {msg['content']}")
            print("\n")
    print("===================================\n")

    rewritten_query = await Query_Rewrite(query, history, runtime.resources.llm_client)

    print("\n===== Rewritten Query =====")
    print(f"原始查询：{query}")
    print(f"改写后的查询：{rewritten_query}")
    print("==============================\n")

    queries = await generate_multi_queries(rewritten_query, runtime.resources.llm_client)

    print("\n===== Multi-Query Generation =====")
    print(f"改写后的查询：{query}")
    print(f"生成的多查询：")
    for q in queries:
        print(q)
    print("=====================================\n")
    
    return {
        "rewritten_query": rewritten_query,
        "multi_queries": queries
    }