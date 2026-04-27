import os
from openai import AsyncAzureOpenAI
from azure.core.credentials import AzureKeyCredential
from core.memory.memory import Memory
from core.rag.Azure_search_index import SearchIndexManager
from core.evaluation.evaluation import OfflineEvaluator, LLMJudgeEvaluator
from config.settings import RuntimeResources


def init_runtime_resources() -> RuntimeResources:

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

    memory = Memory(db_name=db_name)

    # Azure AI Search 服务
    search_index_client = SearchIndexManager(
        endpoint=endpoint,
        credential=credential,
        index_name=index_name,
        dimensions=dimensions,
        model=model,
        embeddings_client=embedding_client,
    )

    # LLM 调用
    llm_client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_AI_KEY"),
        azure_endpoint=os.getenv("AZURE_AI_ENDPOINT"),
        api_version="2024-12-01-preview",
    )

    # 离线评估器
    offline_eval = OfflineEvaluator()
    # 在线评估器
    llm_eval = LLMJudgeEvaluator(llm_client)

    return RuntimeResources(
        llm_client=llm_client,
        search_client=search_index_client,
        memory_store=memory,
        evaluator=llm_eval
    )


    
    
    
    
    
    
    











