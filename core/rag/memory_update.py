import os
from config.settings import ToolRuntime

async def memory_update(query: str, answer: str, runtime: ToolRuntime):
        """
        将当前对话的 session_id, user_query, gpt_response, model 存入数据库
        这个不在 answer_tool 中做是因为防止大模型调用多次，将中途不满意的答案也写入历史记录中，导致后续对话质量下降
        """
        session_id = runtime.session.session_id
        runtime.resources.memory_store.insert(session_id, query, answer, model=os.getenv("AI_MODEL_DEPLOYMENT_NAME"))

        return {
            "memory_update": True
        }