import os
from typing import List, Dict
import json

class OfflineEvaluator:
    """
    离线检索评估器：
    用于评估 RAG Retrieval pipeline 的召回质量
    """
    @staticmethod
    def _norm(doc: Dict) -> str:
        return doc["content"].strip().lower()

    def recall_at_k(self, retrieved_docs: List[Dict], relevant_docs: List[Dict], k: int = 5) -> float:
        """
        Recall@K：
        在所有 relevant docs 中，有多少被成功召回
        """
        if not relevant_docs:
            return 0.0
        
        retrieved_topk = retrieved_docs[: k]

        relevant_set = set(self._norm(d) for d in relevant_docs)
        retrieved_set = set(self._norm(d) for d in retrieved_topk)

        hit_count = len(relevant_set & retrieved_set)

        return hit_count / len(relevant_set)
    
    def precision_at_k(self, retrieved_docs: List[Dict], relevant_docs: List[Dict], k: int = 5) -> float:
        """
        Precision@K:
        Top-K 检索结果中有多少是 relevant docs
        """
        if k == 0:
            return 0.0

        retrieved_topk = retrieved_docs[: k]

        relevant_set = set(self._norm(d) for d in relevant_docs)
        retrieved_set = set(self._norm(d) for d in retrieved_topk)

        hit_count = len(relevant_set & retrieved_set)

        return hit_count / k
    
    def mrr(self, retrieved_docs: List[Dict], relevant_docs: List[Dict]) -> float:
        """
        Mean Reciprocal Rank:
        第一个相关文档出现的位置倒数
        """
        relevant_set = set(self._norm(d) for d in relevant_docs)

        for rank, doc in enumerate(retrieved_docs, start=1):
            if self._norm(doc) in relevant_set:
                return 1.0 / rank

        return 0.0
    
    def evaluate(self, retrieved_docs: List[Dict], relevant_docs: List[Dict], k : int = 5) -> Dict:
        """
        汇总离线评估指标
        """
        return {
            "recall_at_k": self.recall_at_k(retrieved_docs, relevant_docs, k),
            "precision_at_k": self.precision_at_k(retrieved_docs, relevant_docs, k),
            "mrr": self.mrr(retrieved_docs, relevant_docs),
        }

class LLMJudgeEvaluator:
    """
    在线 LLM Judge 评估器：
    用于评估 retrieval relevanve / faithfulness / answer relevance
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.model_name = os.getenv("AI_MODEL_DEPLOYMENT_NAME")

    async def _judge(self, system_prompt: str, user_prompt: str) -> Dict:
        """
        调用 LLM 进行评分，返回 [0, 1]
        """
        response = await self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

        text = response.choices[0].message.content.strip()

        try:
            data = json.loads(text)

            score = float(data.get("score", 0))
            reason = data.get("reason", "")

            score = max(1.0, min(score, 5.0))
            return {
                "score": score / 5.0,
                "reason": reason
            }
        except:
            return {
                "score": 0.0,
                "reason": "parse_failed"
            }
    
    # Retrieval Relevance
    async def judge_retrieval_relevance(self, query: str, docs: List[Dict]) -> float:
        """
        提取的文档是否与用户的问题相关
        """
        context = "\n\n".join(doc["content"] for doc in docs[: 3])

        system_prompt = """
        You are an expert evaluator for document retrieval systems.
        Your task is to assess whether the retrieved documents are relevant to the user's query.

        Score the relevance on a scale of 1 to 5:
        1 = Completely irrelevant
        2 = Mostly irrelevant
        3 = Partially relevant
        4 = Mostly relevant
        5 = Highly relevant

        Return ONLY valid JSON:

        {
        "score": integer from 1 to 5,
        "reason": string explaining why this score was given, otherwise None
        }
        """

        user_prompt = f"""
        User Query:
        {query}

        Retrieved Documents:
        {context}

        Evaluate how relevant the retrieved documents are to the query.
        Return JSON only.
        """

        return await self._judge(system_prompt, user_prompt)
    
    # Faithfulness
    async def judge_faithfulness(self, query: str, docs: List[Dict], answer: str) -> float:
        """
        判断回答由多少是根据提取的文档生成的
        """
        context = "\n\n".join(doc["content"] for doc in docs[: 3])
        system_prompt = """
        You are an expert evaluator for retrieval-augmented generation systems.
        Your task is to determine whether the answer is faithful to the retrieved documents.

        Score the faithfulness on a scale of 1 to 5:
        1 = The answer is unsupported or hallucinated
        2 = The answer is mostly unsupported
        3 = The answer is partially supported
        4 = The answer is mostly supported
        5 = The answer is fully supported by the retrieved documents

        Return ONLY JSON:

        {
        "score": integer from 1 to 5,
        "reason": string explaining why this score was given, otherwise None
        }
        """

        user_prompt = f"""
        User Query:
        {query}

        Retrieved Documents:
        {context}

        Generated Answer:
        {answer}

        Evaluate whether the generated answer is faithful to the retrieved documents.
        Return JSON only.
        """

        return await self._judge(system_prompt, user_prompt)
    
    # Answer Relevance
    async def judge_answer_relevanve(self, query: str, answer: str) -> float:
        """
        判断答案是否回答了用户的问题
        """

        system_prompt = """
        You are an expert evaluator for question answering systems.
        Your task is to determine whether the generated answer adequately addresses the user's question.

        Score the answer relevance on a scale of 1 to 5:
        1 = The answer does not answer the question
        2 = The answer poorly addresses the question
        3 = The answer partially addresses the question
        4 = The answer mostly addresses the question
        5 = The answer fully answers the question

        Return ONLY JSON:

        {
        "score": integer from 1 to 5,
        "reason": string explaining why this score was given, otherwise None
        }
        """

        user_prompt = f"""
        User Query:
        {query}

        Generated Answer:
        {answer}

        Evaluate how well the answer addresses the user's query.
        Return JSON only.
        """

        return await self._judge(system_prompt, user_prompt)
    
    # 汇总评估
    async def evaluate(self, query: str, retrieved_docs: List[Dict], answer: str) -> Dict:
        """
        汇总评估
        """
        retrieval = await self.judge_retrieval_relevance(query, retrieved_docs)
        faithfulness = await self.judge_faithfulness(query, retrieved_docs, answer)
        answer = await self.judge_answer_relevanve(query, answer)

        final_score = (
            0.3 * retrieval["score"] + 
            0.4 * faithfulness["score"] + 
            0.3 * answer["score"]
        )
        
        return {
            "retrieval_relevance": retrieval["score"],
            "retrieval_reason": retrieval["reason"],

            "faithfulness": faithfulness["score"],
            "faithfulness_reason": faithfulness["reason"],

            "answer_relevance": answer["score"],
            "answer_reason": answer["reason"],

            "final_score": final_score
            }