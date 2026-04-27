from typing import List
from config.settings import ToolRuntime

async def critic(query: str, reranked_docs: List[dict], answer: str, runtime: ToolRuntime):
        """
        根据 用户的问题、Reranked_docs 和 answer 进行评判
        retrieval_relevance：提取的文档是否与用户的问题相关
        faithfulness：答案是否忠实于提取的文档
        answer_relevance：答案是否切题，直接回答了用户的问题
        final_score：综合以上维度的最终评分
        """

        llm_eval_result = await runtime.resources.evaluator.evaluate(query, reranked_docs, answer)

        # 找最低分
        stage_scores = {
              "retrieval": llm_eval_result["retrieval_relevance"],
              "faithfulness": llm_eval_result["faithfulness"],
              "answer_relevance": llm_eval_result["answer_relevance"],
        }

        print("\n===== Evaluation Result =====")
        for k, v in stage_scores.items():
            print(f"{k}: {v:.4f}")
        print("================================\n")

        failed_stage = min(stage_scores, key=stage_scores.get)
        if failed_stage == "retrieval":
              reason = llm_eval_result["retrieval_reason"]
        elif failed_stage == "faithfulness":
              reason = llm_eval_result["faithfulness_reason"]
        else: reason = llm_eval_result["answer_reason"]

        return {
            "retrieval_relevance": llm_eval_result["retrieval_relevance"],
            "faithfulness": llm_eval_result["faithfulness"],
            "answer_relevance": llm_eval_result["answer_relevance"],
            "final_score": llm_eval_result["final_score"],
            "reason": reason
        }