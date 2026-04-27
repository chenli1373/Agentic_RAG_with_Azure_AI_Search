import os 
from typing import List 
from config.settings import ToolRuntime 

async def answer(rewritten_query: str, reranked_docs: List[dict], runtime: ToolRuntime): 
    """ 根据检索到的文档生成答案 """ 
    session_id = runtime.session.session_id 
    context = "\n\n".join( f"[Doc {i}]\n{doc['content']}" 
                          for i, doc in enumerate(reranked_docs[:3]) ) 
    # bad case few-shot 
    bad_case_examples = runtime.resources.memory_store.get_bad_cases_for_prompt(session_id, limit=3)
    if not bad_case_examples:
        bad_case_examples = "No bad cases available."

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
    {rewritten_query} 
    
    Retrieved Documents: 
    {context} 
    
    --- 
    
    ### RESPONSE FORMAT 
    Provide a clear and grounded answer based strictly on the documents above. 
    """ 
    
    response = await runtime.resources.llm_client.chat.completions.create(
        model=os.getenv("AI_MODEL_DEPLOYMENT_NAME"), 
        messages=[ 
            {"role": "system", "content": "You are a strict and factual retrieval-based QA assistant."}, 
            {"role": "user", "content": prompt} 
            ] 
        ) 
    
    answer = response.choices[0].message.content 
    
    print("===== AI answer =====") 
    print(answer) 
    print("======================\n") 
    
    return {"answer": answer}