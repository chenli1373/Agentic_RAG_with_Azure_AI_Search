import asyncio
import uuid
import os
import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from azure.core.credentials import AzureKeyCredential

from config.settings import RuntimeResources, SessionContext, ToolRuntime
from core.rag.query_refine import query_refine
from core.rag.retrieval import retrieval
from core.rag.answer import answer
from core.rag.critic import critic
from core.rag.memory_update import memory_update
from services.services import init_runtime_resources

class AgentState(TypedDict):
    query: str

    # query 层
    rewritten_query: str
    multi_queries: List[str]

    # retrieval 层
    retrieval_docs: List[List[dict]]
    RRF_docs: List[dict]
    reranked_docs: List[dict]

    # answer 层
    answer: str

    # ReAct
    scratchpad: List[Dict[str, Any]]
    action: str
    observation: dict

    final_answer: str
    candidate_ready: bool

    done: bool

    rejected: bool
    rejected_reason: bool
    rejected_count: int

    # eval
    retrieval_relevance: float
    faithfulness: float
    answer_relevance: float
    final_score: float

    # control
    retry_count: int

def build_runtime(resources: RuntimeResources, user_id: str) -> ToolRuntime:
    session_id = resources.memory_store.get_or_create_session(user_id)

    history = resources.memory_store.get_history(session_id) or []

    session = SessionContext(
        session_id=session_id,
        history=history
    )

    return ToolRuntime(
        resources=resources,
        session=session
    )

def create_or_get_index_agent(runtime: ToolRuntime):
    async def node(state: AgentState):
        exist = await runtime.resources.search_client.is_index_exists()
        if not exist:
            success = await runtime.resources.search_client.create_index()
            if success:
                print("Index 创建成功！")
        else:
            print("Index 已存在")
            await runtime.resources.search_client.get_index(endpoint=os.getenv("AI_SEARCH_ENDPOINT"),
                                                     credential=AzureKeyCredential(os.getenv("AI_SEARCH_KEY")),
                                                     index_name=os.getenv("INDEX_NAME"))
        return {}
    return node

def upload_documents_agent(runtime: ToolRuntime):
    async def node(state: AgentState):
        sources = [
            "https://learn.microsoft.com/en-us/azure/aks/",
            "https://learn.microsoft.com/en-us/azure/aks/ingress-basic",
            "https://zh.wikivoyage.org/wiki/%E6%AC%A7%E6%B4%B2"
        ]

        await runtime.resources.search_client.upload_new_sources(
            sources=sources,
            memory=runtime.resources.memory_store,
        )
        print("\n文档新增成功！")
        return {}
    return node

def clear_agent(runtime: ToolRuntime):
    async def node(state: AgentState):
        delete_memory = input("是否清除历史记录？(yes / no): ").strip().lower()
        if delete_memory == "yes":
            runtime.resources.memory_store.clear_memory(runtime.session.session_id)
            print("历史记录已清除！")
        
        delete_bad_cases = input("是否清除失败案例？(yes / no): ").strip().lower()
        if delete_bad_cases == "yes":
            runtime.resources.memory_store.clear_bad_case(runtime.session.session_id)
            print("失败案例已清除！")

        return {}
    return node

def reason_agent(runtime: ToolRuntime):
    async def node(state: AgentState):
        query = state["query"]

        scratchpad_text = ""

        for step in state.get("scratchpad", []):
            scratchpad_text += f"""
            Step: {step.get('step')}
            Thought: {step.get('thought')}
            Action: {step.get('action')}
            Observation: {step.get('observation')}
            """
        if not scratchpad_text:
            scratchpad_text = "No tool calls yet."
        
        prompt = f"""
        You are a ReAct-based planning agent inside an Agentic RAG system.

        Your job is NOT to directly solve the question.

        Your job is to decide the NEXT SINGLE STEP in a multi-turn reasoning process.

        You must operate in a loop:
        (Thought → Action → Observation → update state → repeat)

        ---

        ## Core Principle

        At each step:
        You only choose ONE action that best advances the system toward a high-quality final answer.

        You must NOT:
        - produce final answers unless explicitly allowed
        - perform multiple actions
        - assume missing retrieval content
        - skip steps in the pipeline

        ---

        ## Available Actions

        1. query_refine
        Use when:
        - query is ambiguous
        - query can be decomposed
        - retrieval quality is likely to improve with rewrite

        Goal: produce a better search query

        ---

        2. retrieval
        Use when:
        - no documents exist yet
        - existing documents are weak, irrelevant, or incomplete
        - query is already well-formed

        Goal: obtain or improve evidence

        ---

        3. answer
        Use when:
        - relevant documents exist
        - you can synthesize an answer from evidence
        - answer quality is still incomplete or needs refinement
        - You should call "answer" at most ONE time per reasoning chain unless new documents are retrieved.
        Goal: generate or improve answer draft (NOT final unless score is high)

        ---

        ## Finalization Rule (TWO-PATH SYSTEM)

        You may output candidate_ready = True in TWO different cases:

        ---

        ### CASE 1: First-time Finalization (No rejection history)

        Use this ONLY when:
        - final_score ≥ 0.7
        - rejected != True
        - answer is complete and self-contained
        - no further retrieval is needed

        → This is the FIRST attempt at final answer

        ---

        ### CASE 2: Post-Rejection Finalization (Recovery Finalization)

        Use this ONLY when:
        - rejected == True in previous step(s)
        - you have significantly improved the answer
        - previous critic issues are addressed
        - final_score ≥ 0.7

        → This is a RECOVERY after failed attempt

        ---

        ## STRICT RULES

        - If rejected == True:
        you MUST NOT reuse the previous final_answer structure

        - You MUST clearly improve:
        - reasoning depth OR
        - evidence usage OR
        - structure clarity

        - You cannot finalize immediately after a rejection unless improvements are substantial

        ## Critic Feedback Awareness

        If rejected = True:
        - You MUST NOT directly finalize again
        - You must improve the answer first
        - Prefer retrieval or answer refinement
        - Increase reasoning depth before finalization

        If rejection_count >= 1:
        - avoid repeating previous final_answer structure

        ---

        ## Retry Control

        - If retry_count ≥ 2:
        - prefer returning best available answer
        - but still respect quality thresholds

        - Never repeat the same action if the previous observation was insufficient

        ---

        ## Decision Priority (strict order)

        1. If query is unclear → query_refine
        2. Else if no or weak documents → retrieval
        3. Else if documents exist but answer is weak → answer
        4. Else if all scores are high → finalize
        5. Otherwise → improve weakest component

        ---

        ## Scratchpad Usage

        Scratchpad contains full history:
        - previous thoughts
        - actions
        - observations

        You MUST use it to avoid repeating mistakes.

        ---

        ## Output Format (STRICT JSON ONLY)

        If continuing:

        {{
        "thought": "...",
        "action": "query_refine | retrieval | answer",
        "candidate_ready": False
        }}

        If finalizing:

        {{
        "thought": "...",
        "final_answer": "...",
        "candidate_ready": True
        }}

        NO markdown, NO explanation, NO extra text.

        ---

        ## Current Input

        User Query:
        {query}

        State:
        - rewritten_query: {state.get("rewritten_query")}
        - has_docs: {bool(state.get("reranked_docs"))}
        - has_answer: {bool(state.get("answer"))}

        Scores:
        - retrieval_relevance: {state.get("retrieval_relevance")}
        - faithfulness: {state.get("faithfulness")}
        - answer_relevance: {state.get("answer_relevance")}
        - final_score: {state.get("final_score")}
        - retry_count: {state.get("retry_count", 0)}

        Scratchpad:
        {scratchpad_text}
        """

        response = await runtime.resources.llm_client.chat.completions.create(
            model=os.getenv("AI_MODEL_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are a helpful and precise assistant for answering questions."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        result = json.loads(response.choices[0].message.content)

        scratchpad = state.get("scratchpad", [])
        candidate_ready = result.get("candidate_ready", False)
        
        # ========== 模型认为答案已经准备好 ==========
        if candidate_ready:
            scratchpad.append({
                "step": len(scratchpad),
                "thought": result["thought"],
                "action": "finalize_answer",
                "observation": result.get("final_answer", "")
            })

            print("\n===== Reasoning Step =====")
            print(f"Step: {len(scratchpad) - 1}")
            print(f"Thought: {result['thought']}")
            print(f"Action: finalize_answer")
            print(f"Candidate Answer: {result.get('final_answer', '')}")
            print("Done: candidate_ready=True")
            print("============================\n")

            return {
                "final_answer": result.get("final_answer", ""),
                "candidate_ready": True,
                "scratchpad": scratchpad
            }

        # ========== 模型决定继续调用工具 ==========
        scratchpad.append({
            "step": len(scratchpad),
            "thought": result["thought"],
            "action": result["action"],
            "observation": None
        })

        print("\n===== Reasoning Step =====")
        print(f"Step: {len(scratchpad) - 1}")
        print(f"Thought: {result['thought']}")
        print(f"Action: {result['action']}")
        print("Done: False")
        print("============================\n")

        return {
            "scratchpad": scratchpad,
            "action": result.get("action"),
            "candidate_ready": False
        }
    return node

def is_done(state: AgentState):
    if state["candidate_ready"] and not state.get("rejected", False):
        return "critic"
    return "tool"

def tool_executor(runtime: ToolRuntime):
    async def node(state: AgentState):
        action = state["action"]

        # 解决数据一致性
        query = state.get("rewritten_query") or state["query"]
        multi_queries = state.get("multi_queries") or [query]

        updates = {}

        if action == "query_refine":
            result = await query_refine(
                query=query,
                runtime=runtime
            )

            updates.update({
                "rewritten_query": result["rewritten_query"],
                "multi_queries": result["multi_queries"]
            })

            observation = result
        
        elif action == "retrieval":
            result = await retrieval(
                rewritten_query=query,
                multi_queries=multi_queries,
                runtime=runtime
            )

            updates.update({
                "retrieval_docs": result["retrieval_docs"],
                "RRF_docs": result["RRF_docs"],
                "reranked_docs": result["reranked_docs"]
            })

            observation = result

        elif action == "answer":
            docs = state.get("reranked_docs", [])
            result = await answer(
                rewritten_query=query,
                reranked_docs=docs,
                runtime=runtime
            )

            updates.update({
                "answer": result["answer"]
            })

            observation = result
        
        else:
            raise ValueError(f"Unknown action: {action}")
        
        # ReAct trace
        scratchpad = state.get("scratchpad", [])

        if scratchpad:
            scratchpad[-1]["observation"] = {
                "tool": action,
                "result": observation
            }
        
        updates.update({
            "scratchpad": scratchpad,
            "observation": observation
        })

        return updates
    return node

def critic_agent(runtime: ToolRuntime):
    async def node(state: AgentState):
        reranked_docs = state.get("reranked_docs", [])
        answer = state.get("answer", "") or state.get("final_answer", "")

        query = state["query"]

        result = await critic(
            query=query,
            reranked_docs=reranked_docs,
            answer=answer,
            runtime=runtime
        )

        updates = {
            "retrieval_relevance": result["retrieval_relevance"],
            "faithfulness": result["faithfulness"],
            "answer_relevance": result["answer_relevance"],
            "final_score": result["final_score"],
            "observation": result
        }

        # 只输出信号，不控制
        if state.get("candidate_ready") and result["final_score"] <= 0.7:
            updates["rejected"] = True
            updates["rejection_reason"] = result.get("reason", "")
            updates["rejection_count"] = state.get("rejection_count", 0) + 1
        else: 
            updates["rejected"] = False
            updates["rejection_reason"] = ""

        # 只有当失败时才计数
        if result["final_score"] < 0.7:
            updates["retry_count"] = state.get("retry_count", 0) + 1
        
        scratchpad = state.get("scratchpad", [])
        scratchpad.append({
            "step": len(scratchpad),
            "thought": f"Critic score={result['final_score']:.2f}, decide whether to retry",
            "action": "critic",
            "observation": result
        })
            
        updates["scratchpad"] = scratchpad

        return updates
    return node

def route_after_critic(state: AgentState):
    if state.get("candidate_ready") and not state.get("rejected", False):
        return "memory_update"
    else: return "bad_cases_save"

def bad_cases_agent(runtime: ToolRuntime):
    async def node(state: AgentState):
        # 只有 rejected 才记录
        if not state.get("rejected", False):
            return {}
        
        runtime.resources.memory_store.save_bad_cases(
            session_id=runtime.session.session_id,
            stage="critic",
            query=state["query"],
            input_data={
                "reranked_docs": state.get("reranked_docs", []),
                "answer": state.get("answer", ""),
                "scratchpad": state.get("scratchpad", [])
            },
            output_data={
                "final_score": state.get("final_score"),
                "rejection_reason": state.get("rejected_reason")
            },
            score=state.get("final_score", 0.0),
            reason=state.get("rejected_reason", "")
        )

        return {}
    return node

def memory_update_agent(runtime: ToolRuntime):
    async def node(state: AgentState):
        query = state["query"]
        answer = state.get("final_answer", "") or state.get("answer", "")

        await memory_update(
            query=query,
            answer=answer,
            runtime=runtime
        )

        print("历史信息已记录.")

        return {"done": True}
    return node

def build_graph(runtime: ToolRuntime):

    graph = StateGraph(AgentState)

    graph.add_node("index", create_or_get_index_agent(runtime))
    graph.add_node("upload", upload_documents_agent(runtime))
    graph.add_node("clear", clear_agent(runtime))
    graph.add_node("reason", reason_agent(runtime))
    graph.add_node("tool_executor", tool_executor(runtime))
    graph.add_node("critic", critic_agent(runtime))
    graph.add_node("bad_cases", bad_cases_agent(runtime))
    graph.add_node("memory_update", memory_update_agent(runtime))

    graph.set_entry_point("index")
    graph.add_edge("index", "upload")
    graph.add_edge("upload", "clear")
    graph.add_edge("clear", "reason")
    graph.add_conditional_edges(
        "reason",
        is_done,
        {
            "tool": "tool_executor",
            "critic": "critic"
        }
    )

    graph.add_edge("tool_executor", "reason")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "bad_cases_save": "bad_cases",
            "memory_update": "memory_update"
        }
    )
    graph.add_edge("bad_cases", "reason")
    graph.add_edge("memory_update", END)

    return graph.compile()

async def main(is_render: bool = False):
    resources = init_runtime_resources()
    print("Runtime resources 初始化成功！")

    user_id = input("请输入用户ID: ").strip()
    runtime = build_runtime(resources, user_id)
    print("Tool runtime 构建成功！")

    try:
        graph = build_graph(runtime)
        print("graph 构建成功！")

        if is_render:
            png_data = graph.get_graph().draw_png()
            with open("rag_graph.png", "wb") as f:
                f.write(png_data)

        query = input("请输入您的问题: ").strip()

        result = await graph.ainvoke({
            "query": query,
            "scratchpad": []
        })
    
    finally:
        await resources.search_client.close()
        print("资源已释放！")

    print("\n===== Final Answer =====")
    print(result["final_answer"])

if __name__ == "__main__":
    is_render = True
    asyncio.run(main(is_render))
