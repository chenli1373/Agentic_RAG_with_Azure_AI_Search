from typing import Optional, List

import os
import json

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchIndex,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
)
# from azure.ai.inference.aio import EmbeddingsClient
from openai import AsyncAzureOpenAI
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

class SearchIndexManager:
    """
    这个类为用户的查询搜索上下文
    """
    
    # 一行中不同字符的个数
    MIN_DIFF_CHARACTERS_IN_LINE = 5
    # 一行长度
    MIN_LINE_LENGTH = 5

    def __init__(self,
                 endpoint: str,
                 credential: AzureKeyCredential,
                 index_name: str,
                 dimensions: Optional[int],
                 model: str,
                 embeddings_client: AsyncAzureOpenAI
                 ) -> None:
        self._endpoint = endpoint
        self._credential = credential
        self._index_name = index_name
        self._dimensions = dimensions
        self._model = model
        self._embeddings_client = embeddings_client
        self._index = None
        self._client = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

    # 如果没有服务器，则连接搜索服务器
    def _get_client(self):
        if self._client is None:
            self._client = SearchClient(
                endpoint=self._endpoint, index_name=self._index_name, credential=self._credential
            )
        return self._client
    
    # 根据用户问题查询
    async def search(self, query) -> List[dict]:
        self._raise_if_no_index()

        emb = await self._embeddings_client.embeddings.create(
            input=query,
            model=self._model
        )

        vector = emb.data[0].embedding

        vector_query = VectorizedQuery(
           vector=vector,
           k_nearest_neighbors=5,
           fields="embedding"
        )
        
        response = await self._get_client().search(
            vector_queries=[vector_query],
            select=['content', 'metadata']
        )

        results = []
        async for result in response:
            results.append({
                "content": result["content"],
                "metadata": result.get("metadata"),
                "score": result.get("@search.score", 0.0)
            })

        return results

    def load_and_split_document(self, sources: List[str]) -> List[Document]:
        all_documents = []

        for source in sources:
            # 判断是否为 URL
            if source.startswith("http://") or source.startswith("https://"):
                loader = WebBaseLoader(source)
            
            # 判断本地文件类型
            elif source.endswith(".pdf"):
                loader = PyPDFLoader(source)
            
            elif source.endswith(".docx"):
                loader = Docx2txtLoader(source)
            
            elif source.endswith(".html"):
                loader = UnstructuredHTMLLoader(source)
            
            else: ValueError(f"Unsupported source type: {source}")

            docs = loader.load()
            
            for doc in docs:
                doc.metadata["source"] = source

            all_documents.extend(docs)
        
        return self.text_splitter.split_documents(all_documents)

    # 根据上一个函数创建的 embedding_file 上传文档
    async def upload_documents(self, documents: List[Document]) -> None:
        self._raise_if_no_index()

        texts = []
        metadata = []

        for doc in documents:
            text = doc.page_content
            if not text:
                continue

            texts.append(text)
            metadata.append(doc.metadata)
        
        batch_size = 2000
        upload_batch = []
        index = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            response = (await self._embeddings_client.embeddings.create(
                input=batch_texts,
                model=self._model
            ))

            embedding = [d.embedding for d in response.data]

            for text, emb, meta in zip(batch_texts, embedding, metadata[i : i + batch_size]):
                upload_batch.append({
                    "embedId": str(index),
                    "content": text,
                    "embedding": emb,
                    "metadata": json.dumps(meta)
                })
                index += 1
        
        await self._get_client().upload_documents(upload_batch)
    
    # 上传新的 source
    async def upload_new_sources(self, sources: List[str], memory):
        for source in sources:

            if memory.is_source_indexed(source):
                print(f"跳过已经上传的文件：{source}")
                continue

            docs = self.load_and_split_document([source])
            await self.upload_documents(docs)
            memory.mark_source_indexed(source)
            print(f"新增上传完成：{source}")
    
    # index 是否为空
    async def is_index_empty(self) -> bool:
        if self._index is None:
            raise ValueError("Unable to perform the operation as the index is absent. "
                            "To create index please call create_index")
        
        document_count = await self._get_client().get_document_count()
        return document_count == 0
    
    def _raise_if_no_index(self) -> None:
        if self._index is None:
            raise ValueError("Unable to perform the operation as the index is absent."
                             "To create index please call create_index")
    
    # 删除 index
    async def delete_index(self):
        self._raise_if_no_index()
        async with SearchIndexClient(endpoint=self._endpoint, credential=self._credential) as ix_client:
            await ix_client.delete_index(self._index_name)
        self._index = None
    
    # 检查维度是否一致
    def _check_dimensions(self, vector_index_dimensions: Optional[int] = None) -> int:
        if vector_index_dimensions is None:
            if self._dimensions is None:
                raise ValueError("No embedding dimensions were provided in neither dimensions in the constructor nor in vector_index_dimensions"
                    "Dimensions are needed to build the search index, please provide the vector_index_dimensions.")
            vector_index_dimensions = self._dimensions
        
        if self._dimensions is not None and vector_index_dimensions != self._dimensions:
            raise ValueError("vector_index_dimensions is different from dimensions provided to constructor.")
        
        return vector_index_dimensions
    
    # 确认 index 已经创建
    # ensure_index_created() -> _check_dimensions() -> get_or_create_index()
    async def is_index_exists(self) -> bool:
        async with SearchIndexClient(endpoint=self._endpoint, credential=self._credential) as client:
            try:
                await client.get_index(self._index_name)
                print("index 已存在，无需创建！")
                return True
            except ResourceNotFoundError:
                print("index 不存在！")
                return False
    
    # 如果 index_name 对应的 index 存在，获取这个 index
    # 如果不存在，则创建
    # get_or_create_index() -> _index_create()
    async def get_index(self, endpoint: str, credential: AzureKeyCredential, index_name: str) -> None:
        async with SearchIndexClient(endpoint=endpoint, credential=credential) as ix_client:
            try:
                self._index = await ix_client.get_index(index_name)
                # print("index 已存在，无需创建.")
            except ResourceNotFoundError:
                pass

    # 外部调用函数，创建 index 前，会检查维度是否一致
    # create_index() -> _check_dimensions() -> _index_create()
    async def create_index(self, vector_index_dimensions: Optional[int] = None) -> bool:
        vector_index_dimensions = self._check_dimensions(vector_index_dimensions)
        try:
            self._index = await SearchIndexManager._index_create(endpoint=self._endpoint, credential=self._credential, index_name=self._index_name, dimensions=vector_index_dimensions)
            return True
        except HttpResponseError as e:
            print("Azure create_index failed:")
            print(e)
            return False
    
    # 内部调用函数，创建索引，最底层的函数
    @staticmethod
    async def _index_create(endpoint: str, credential: AzureKeyCredential, index_name: str, dimensions: int) -> SearchIndex:
        async with SearchIndexClient(endpoint=endpoint, credential=credential) as ix_client:
            fields = [
                SimpleField(name="embedId", type=SearchFieldDataType.String, key=True),
                SearchField(
                    name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    vector_search_dimensions=dimensions,
                    searchable=True,
                    vector_search_profile_name="embedding_config"
                ),
                SimpleField(name="content", type=SearchFieldDataType.String, hidden=False),
                SimpleField(name="metadata", type=SearchFieldDataType.String, filterable=True, searchable=True)
            ]
            vector_search = VectorSearch(
                profiles=[VectorSearchProfile(name="embedding_config", algorithm_configuration_name="embed-algorithms-config")],
                algorithms=[HnswAlgorithmConfiguration(name="embed-algorithms-config")],
            )
            search_index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
            new_index = await ix_client.create_index(search_index)
        return new_index
        
    async def close(self):
        if self._client:
            await self._client.close()
        
        if self._embeddings_client:
            await self._embeddings_client.close()

# async def main():
#     endpoint = os.getenv("AI_SEARCH_ENDPOINT")
#     model = os.getenv("AI_EMBEDDING_DEPLOYMENT_NAME")
#     embedding_client = AsyncAzureOpenAI(
#         azure_endpoint=os.getenv("AZURE_AI_ENDPOINT_EMBEDDINGS"),
#         api_key=os.getenv("AZURE_AI_KEY"),
#         api_version="2024-02-01"
#     )
#     credential = AzureKeyCredential(os.getenv("AI_SEARCH_KEY"))
#     dimensions = int(os.getenv("AI_EMBED_DIMENSION"))
#     index_name = os.getenv("INDEX_NAME")

#     file_path = "./data/1.pdf"

#     search_index = SearchIndexManager(
#         endpoint=endpoint,
#         credential=credential,
#         index_name=index_name,
#         dimensions=dimensions,
#         model=model,
#         embeddings_client=embedding_client
#     )

#     print("初始化成功！")

#     success = await search_index.create_index()
#     print(success)

#     if success:
#         print("索引创建成功！")

#         docs = search_index.load_and_split_document(file_path)
#         print("文件加载成功！")

#         await search_index.upload_documents(docs)
#         print("文件上传成功！")

#         result = await search_index.search("这个文档是关于什么的")
#         print(result)
    
#     await search_index.close()

# if __name__ == "__main__":
#     asyncio.run(main())