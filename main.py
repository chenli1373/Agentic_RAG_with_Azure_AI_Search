from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List
from azure.core.credentials import AzureKeyCredential
from openai import AsyncAzureOpenAI
from Azure_search_index import SearchIndexManager
from memory import Memory
from langchain_utils import (
    Query_Rewrite, 
    generate_multi_queries, 
    reciprocal_rank_fusion, 
    rerank,
    build_relevant_docs_from_answer,
    QAIterator
)
from evaluation import OfflineEvaluator, LLMJudgeEvaluator
import os 
import asyncio
from dotenv import load_dotenv

load_dotenv()


class EvalState(TypedDict, total=False):
    recall_at_k: float
    precision_at_k: float
    mrr: float
    answer_quality: float
    final_offline_eval_score: float

    retrieval_judge: float
    faithfulness: float
    answer_relevance: float
    final_LLM_Judge_score: float

class RAGState(TypedDict):
    user_id: str
    session_id: str
    query: str
    rewritten_query: str
    queries: List[str]
    history: List[dict]
    retrieval_docs: List[List[dict]]
    RRF_docs: List[dict]
    reranked_docs: List[dict]
    answer: str

    # evaluation
    is_offline: bool
    next: str
    eval: EvalState
    done: bool

class Services:
    def __init__(self):
        endpoint = os.getenv("AI_SEARCH_ENDPOINT")
        model = os.getenv("AI_EMBEDDING_DEPLOYMENT_NAME")
        # embedding 调用
        embedding_client = AsyncAzureOpenAI(
            azure_endpoint=os.getenv("AZURE_AI_ENDPOINT_EMBEDDINGS"),
            api_key=os.getenv("AZURE_AI_KEY"),
            api_version="2024-02-01"
        )
        credential = AzureKeyCredential(os.getenv("AI_SEARCH_KEY"))
        dimensions = int(os.getenv("AI_EMBED_DIMENSION"))
        index_name = os.getenv("INDEX_NAME")

        db_name = os.getenv("DB_NAME")

        self.memory = Memory(db_name=db_name)

        # Azure AI Search 服务
        self.search_index_client = SearchIndexManager(
            endpoint=endpoint,
            credential=credential,
            index_name=index_name,
            dimensions=dimensions,
            model=model,
            embeddings_client=embedding_client,
        )

        # LLM 调用
        self.llm_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_AI_KEY"),
            azure_endpoint=os.getenv("AZURE_AI_ENDPOINT"),
            api_version="2024-12-01-preview",
        )

        # 离线评估器
        self.offline_eval = OfflineEvaluator()
        # 在线评估器
        self.llm_eval = LLMJudgeEvaluator(self.llm_client)

        # qa 数据集
        qa_path = "./QA.json"
        self.qa_iterator = QAIterator(qa_path)

def session_init_agent(services: Services):
    async def node(state: RAGState):
        session_id = services.memory.get_or_create_session(state["user_id"])
        return {"session_id": session_id}
    return node

def create_or_get_index_agent(services: Services):
    async def node(state: RAGState):
        exist = await services.search_index_client.is_index_exists()
        if not exist:
            success = await services.search_index_client.create_index()
            if success:
                print("index 创建成功！")
        else: 
            print("index 已存在")
            await services.search_index_client.get_index(endpoint=os.getenv("AI_SEARCH_ENDPOINT"),
                                                     credential=AzureKeyCredential(os.getenv("AI_SEARCH_KEY")),
                                                     index_name=os.getenv("INDEX_NAME"))
    return node

def upload_documents_agent(services: Services):
    async def node(state: RAGState):
        sources = [
            "https://learn.microsoft.com/en-us/azure/aks/",
            "https://learn.microsoft.com/en-us/azure/aks/ingress-basic",
            "https://zh.wikivoyage.org/wiki/%E6%AC%A7%E6%B4%B2"
        ]
        
        await services.search_index_client.upload_new_sources(
            sources=sources,
            memory=services.memory
        )
        print("\n文档新增完成！")
        return {}
    return node

def clear_agent(services: Services):
    async def node(state: RAGState):
        delete = input("是否要清除历史对话记录？(yes / no): ")

        if delete == "yes":
            services.memory.clear_memory(session_id=state["session_id"])
            print("历史消息清除成功！")

            services.memory.clear_bad_case(session_id=state["session_id"])
            print("bad case库清除成功！")
        return {}
    return node

def is_offline_eval(services: Services):
    async def node(state: RAGState):
        offline = input("是否使用离线评估器？(yes / no): ")
        is_offline = offline.lower() == "yes"

        return {
            "is_offline": is_offline,
            "next": "qa_input" if is_offline else "user_input"
        }
    return node

def is_offline_router(state: RAGState):
    return state.get("next", "user_input")

def qa_input_agent(services: Services):
    async def node(state: RAGState):
        if not services.qa_iterator.has_next():
            return {"done": True}
        
        qa_item = services.qa_iterator.next()
        query = qa_item["question"]

        return {"query": query, "done": False}
    return node

def is_done_agent(services: Services):
    async def node(state: RAGState):
        done = state["done"]
        if done == True:
            return "exit"
        else: return "continue"
    return node

def user_input_agent(services: Services):
    async def node(state: RAGState):
        user_id = state["user_id"]

        query = input(f"{user_id}, Enter your question or 'exit' to quit: ").strip()

        while query != 'exit':
            if not query:
                print("No query entered. Please try again.")
                query = input(f"{user_id}, Enter your question or 'exit' to quit: ").strip()
            else: break
        if query == 'exit':
            print("Exiting the Agentic RAG System. GoodBye!")
            exit(0)
        return {"query": query}
    return node

def query_rewrite_agent(services: Services):
    async def node(state: RAGState):
        query = state["query"]

        history = services.memory.get_history(state["session_id"]) or []

        print("\n===== Conversation History =====")
        if not history:
            print("There is no history yet.")
        else:
            for msg in history:
                print(f"{msg['role']}: {msg['content']}")
                print("\n")
        print("===================================\n")

        rewritten = await Query_Rewrite(query, history, services.llm_client)

        print("\n===== Query Rewrite =====")
        print(f"原查询：{query}")
        print(f"改写后的查询：{rewritten}")
        print("============================\n")

        return {"rewritten_query": rewritten, "history": history}
    return node

def multi_query_agent(services: Services):
    async def node(state: RAGState):
        query = state["rewritten_query"]

        queries = await generate_multi_queries(query, services.llm_client)

        print("\n===== Multi-Query Generation =====")
        print(f"改写后的查询：{query}")
        print(f"生成的多查询：")
        for q in queries:
            print(q)
        print("=====================================\n")

        return {"queries": queries}
    return node

def retrieval_agent(services: Services):
    async def node(state: RAGState):
        queries = state["queries"]

        tasks = [
            services.search_index_client.search(q)
            for q in queries
        ]

        results = await asyncio.gather(*tasks)

        print("\n===== retrieval results =====")
        for i, r in enumerate(results):
            print(f"\n--- query {i} ---")
            print(r)
        print("=================================\n")

        return {"retrieval_docs": results}
    return node

def fusion_agent(services: Services):
    async def node(state: RAGState):
        retrieval_docs = state["retrieval_docs"]
        fused_docs = reciprocal_rank_fusion(retrieval_docs)

        print("\n===== RRF Fusion =====")
        for i, doc in enumerate(fused_docs):
            print(f"\n[{i}]")
            print(doc)
        print("=========================\n")

        return {"RRF_docs": fused_docs}
    return node

def rerank_agent(services: Services):
    async def node(state: RAGState):
        query = state["rewritten_query"]
        RRF_docs = state["RRF_docs"]

        reranked_docs = rerank(query, RRF_docs, top_k=5)

        print("\n===== Reranked Result =====")
        for i, doc in enumerate(reranked_docs):
            score_preview = doc.get("score", None) if isinstance(doc, dict) else None
            print(f"\n[{i}] score={score_preview}")
            print(doc)
        print("==============================\n")

        return {"reranked_docs": reranked_docs}
    return node

def answer_agent(services: Services):
    async def node(state: RAGState):
        query = state["rewritten_query"]
        reranked_docs = state["reranked_docs"]

        context = "\n\n".join(
            f"[Doc {i}]\n{doc['content']}"
            for i, doc in enumerate(reranked_docs[:3])
        )

        # bad case few-shot
        bad_case_examples = services.memory.get_bad_cases_for_prompt(
            session_id=state["session_id"],
            limit=3
        )

        prompt = f"""You are a highly reliable retrieval-based question answering assistant.

        ### CORE RULES
        - Use ONLY the retrieved documents to answer.
        - Do NOT hallucinate or use external knowledge.
        - If information is insufficient, respond: "I don't know".
        - Be concise, factual, and grounded.

        ---

        ### LEARNING FROM PAST FAILURES (DO NOT COPY)

        The following are previous failure cases. You must learn from them and avoid repeating the same mistakes.

        {bad_case_examples}

        ---

        ### CURRENT TASK

        User Question:
        {query}

        Retrieved Documents:
        {context}

        ---

        ### RESPONSE FORMAT
        Provide a clear and grounded answer based strictly on the documents above.
        """
        response = await services.llm_client.chat.completions.create(
            model=os.getenv("AI_MODEL_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are a strict and factual retrieval-based QA assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content

        services.memory.insert(state["session_id"], state["query"], answer, os.getenv("AI_MODEL_DEPLOYMENT_NAME"))
        print("===== AI answer =====")
        print(answer)
        print("======================\n")

        return {"answer": answer}
    return node

def should_offline(services: Services):
    async def node(state: RAGState):
        is_offline = state["is_offline"]
        return "offline_eval" if is_offline else "LLM_Judge"
    return node

def Offline_eval_agent(services: Services):
    async def node(state: RAGState):
        query = state["rewritten_query"]
        qa_item = services.qa_iterator.next()
        ground_truth = qa_item["answer"]
        retrieval_docs = state["reranked_docs"]
        answer = state["answer"]

        if not retrieval_docs:
            state["eval"] = {
                "recall_at_k": 0.0,
                "precision_at_k": 0.0,
                "mrr": 0.0,
                "final_eval_score": 0.0,
            }
            return state

        relevant_docs = build_relevant_docs_from_answer(ground_truth)

        # 离线评估指标
        metrics = services.offline_eval.evaluate(
            retrieved_docs=retrieval_docs,
            relevant_docs=relevant_docs,
            k=5
        )

        # LLM Judge - Answer Quality
        system_prompt = """
        You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.

        Your task is to evaluate how well the generated answer matches the ground truth answer.

        Scoring rules (0-10):
        - 0-2: completely wrong or irrelevant
        - 3-4: partially related but mostly incorrect
        - 5-6: partially correct, missing key points
        - 7-8: mostly correct with minor omissions
        - 9-10: fully correct and complete

        IMPORTANT:
        - Only output a single number (0-10)
        - No explanation
        """

        user_prompt = f"""
        User Question:
        {query}

        Ground Truth Answer:
        {ground_truth}

        Generated Answer:
        {answer}

        Please evaluate the quality of the generated answer.
        """

        response = await services.llm_client.chat.completions.create(
            model=os.getenv("AI_MODEL_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

        try:
            answer_score = float(response.choices[0].message.content.strip())
            answer_score = max(0.0, min(10.0, answer_score))
            answer_score = answer_score / 10.0
        except:
            answer_score = 0.0
        
        final_score = (
            0.4 * metrics["recall_at_k"] +
            0.3 * metrics["precision_at_k"] +
            0.3 * answer_score
        )

        state["eval"] = {
            "recall_at_k": metrics["recall_at_k"],
            "precision_at_k": metrics["precision_at_k"],
            "mrr": metrics["mrr"],
            "answer_quality": answer_score,
            "final_offline_eval_score": final_score
        }

        print("\n===== Offline Eval =====")
        print(f"Query: {query}")
        print(f"Recall@5: {metrics['recall_at_k']:.3f}")
        print(f"Precision@5: {metrics['precision_at_k']:.3f}")
        print(f"Answer Quality: {answer_score:.3f}")
        print(f"Final Offline Score: {final_score:.3f}")
        print("=================================\n")
    return node

def LLM_Judge_agent(services: Services):
    async def node(state: RAGState):
        query = state["query"]
        docs = state["reranked_docs"]
        answer = state["answer"]
        LLM_eval_result = await services.llm_eval.evaluate(query, docs, answer)

        eval = {
            "retrieval_judge": LLM_eval_result["retrieval_relevance"],
            "faithfulness": LLM_eval_result["faithfulness"],
            "answer_relevance": LLM_eval_result["answer_relevance"],
            "final_LLM_Judge_score": LLM_eval_result["final_score"]
        }

        print("\n===== Evaluation Result =====")
        for k, v in eval.items():
            print(f"{k}: {v:.4f}")
        print("================================\n")
        return {"eval": eval}
    return node
        
def bad_case_agent(services: Services, threshold: float = 0.7):
    async def node(state: RAGState):
        eval_result = state["eval"]
        final_score = eval_result["final_LLM_Judge_score"]

        # 判断是否是 bad case
        if final_score >= threshold:
            print("评估通过，这不是一个 bad case.\n")
            return {}
        
        # 找最低分
        stage_scores = {
            "retrieval": eval_result["retrieval_judge"],
            "faithfulness": eval_result["faithfulness"],
            "answer_relevance": eval_result["answer_relevance"]
        }

        failed_stage = min(stage_scores, key=stage_scores.get)
        failed_score = stage_scores[failed_stage]

        query = state["query"]

        if failed_stage == "retrieval":
            input_data = query
            output_data = str(state["reranked_docs"][: 3])
            reason = (
                f"Retrieved documents are not sufficiently relevant to the user query. "
                f"retrieval_judge={failed_score:.4f}. "
                f"Possible causes: "
                f"(1) query rewrite changed the user intent incorrectly; "
                f"(2) multi-query generation introduced noisy queries; "
                f"(3) vector retrieval returned semantically unrelated chunks."
            )
        elif failed_stage == "faithfulness":
            input_data = str(state["reranked_docs"][: 3])
            output_data = state["answer"]
            reason = (
                f"The generated answer is not well supported by the retrieved documents. "
                f"faithfulness={failed_score:.4f}. "
                f"Possible causes: "
                f"(1) LLM hallucinated information not present in context; "
                f"(2) prompt instructions were too weak to constrain the answer; "
                f"(3) retrieved context lacked enough evidence."
            )
        else: # answer relevance
            input_data = query
            output_data = state["answer"]
            reason = (
                f"The generated answer does not adequately address the user's question. "
                f"answer_relevance={failed_score:.4f}. "
                f"Possible causes: "
                f"(1) answer focused on irrelevant details; "
                f"(2) prompt failed to emphasize user intent; "
                f"(3) retrieved documents only partially covered the question."
            )
        
        # 保存 bad case
        services.memory.save_bad_cases(
            session_id=state["session_id"],
            stage=failed_stage,
            query=query,
            input_data=input_data,
            output_data=output_data,
            score=final_score,
            reason=reason
        )

        print("===== BAD CASE DETECTED =====")
        print(f"Failed Score: {final_score:.4f}")
        print(f"Failed Stage: {failed_stage}")
        print(f"Reason: {reason}")
        print("==============================\n")
        return {}
    return node

def continue_offline(services: Services):
    async def node(state: RAGState):
        is_offline = state["is_offline"]
        if is_offline:
            return "continue"
        else: return "user_model"
    return node

def continue_check_agent(services: Services):
    async def node(state: RAGState):
        cont = input("Do you want to ask another question? (yes/no):").strip().lower()
        if cont == "yes":
            return "continue"
        else:
            print("Thank you for using the Agentic RAG System. GoodBye")
            return "exit"
    return node

def build_graph(services: Services):
    graph = StateGraph(RAGState)

    # 添加节点
    graph.add_node("session_init", session_init_agent(services))
    graph.add_node("create_index", create_or_get_index_agent(services))
    graph.add_node("upload_documents", upload_documents_agent(services))
    graph.add_node("clear_memory", clear_agent(services))
    graph.add_node("is_offline", is_offline_eval(services))
    graph.add_node("qa_input", qa_input_agent(services))
    graph.add_node("is_done", lambda state : state)
    graph.add_node("user_input", user_input_agent(services))
    graph.add_node("query_rewrite", query_rewrite_agent(services))
    graph.add_node("multi_query", multi_query_agent(services))
    graph.add_node("retrieval", retrieval_agent(services))
    graph.add_node("RRF", fusion_agent(services))
    graph.add_node("rerank", rerank_agent(services))
    graph.add_node("answer", answer_agent(services))
    graph.add_node("should_offline", lambda state : state)
    graph.add_node("offline_eval", Offline_eval_agent(services))
    graph.add_node("LLM_Judge", LLM_Judge_agent(services))
    graph.add_node("bad_case", bad_case_agent(services))
    graph.add_node("continue_offline", lambda state : state)
    graph.add_node("router", lambda state : state)

    # 添加边
    graph.set_entry_point("session_init")
    graph.add_edge("session_init", "create_index")
    graph.add_edge("create_index", "upload_documents")
    graph.add_edge("upload_documents", "clear_memory")
    graph.add_edge("clear_memory", "is_offline")
    graph.add_conditional_edges(
        "is_offline",
        is_offline_router,
        {
            "qa_input": "qa_input",
            "user_input": "user_input"
        }
    )
    graph.add_edge("qa_input", "is_done")
    graph.add_conditional_edges(
        "is_done",
        is_done_agent(services),
        {
            "continue": "query_rewrite",
            "exit": END
        }
    )
    graph.add_edge("user_input", "query_rewrite")
    graph.add_edge("query_rewrite", "multi_query")
    graph.add_edge("multi_query", "retrieval")
    graph.add_edge("retrieval", "RRF")
    graph.add_edge("RRF", "rerank")
    graph.add_edge("rerank", "answer")
    graph.add_edge("answer", "should_offline")
    graph.add_conditional_edges(
        "should_offline",
        should_offline(services),
        {
            "offline_eval": "offline_eval",
            "LLM_Judge": "LLM_Judge"
        }
    )
    graph.add_edge("offline_eval", "LLM_Judge")
    graph.add_edge("LLM_Judge", "bad_case")
    graph.add_edge("bad_case", "continue_offline")
    graph.add_conditional_edges(
        "continue_offline",
        continue_offline(services),
        {
            "continue": "qa_input",
            "user_model": "router"
        }
    )
    graph.add_conditional_edges(
        "router",
        continue_check_agent(services),
        {
            "continue": "clear_memory",
            "exit": END
        }
    )

    return graph.compile()

async def main(render: bool = False):
    services = Services()
    print("基础服务已经创建完成！")

    try:
        graph = build_graph(services)
        print("graph创建完毕")

        if render:
            png_data = graph.get_graph().draw_png()
            with open("rag_graph.png", "wb") as f:
                f.write(png_data)

        user_id = input("请输入用户ID：").strip()

        await graph.ainvoke({
            "user_id": user_id
        })
    
    finally:
        await services.search_index_client.close()
        print("资源已释放")

if __name__ == '__main__':
    is_render = True
    asyncio.run(main(is_render))