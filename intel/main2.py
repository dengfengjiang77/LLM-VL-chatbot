# 设置OpenMP线程数为8
import os
import time
os.environ["OMP_NUM_THREADS"] = "8"

import torch
from typing import Any, List, Optional
import gradio as gr


# 从llama_index库导入HuggingFaceEmbedding类，用于将文本转换为向量表示
from llama_index.core.embeddings import HuggingFaceEmbedding
# 从llama_index库导入ChromaVectorStore类，用于高效存储和检索向量数据
from llama_index.core.embeddings import ChromaVectorStore
# 从llama_index库导入PyMuPDFReader类，用于读取和解析PDF文件内容
from llama_index.readers.file import PyMuPDFReader
# 从llama_index库导入NodeWithScore和TextNode类
# NodeWithScore: 表示带有相关性分数的节点，用于排序检索结果
# TextNode: 表示文本块，是索引和检索的基本单位。节点存储文本内容及其元数据，便于构建知识图谱和语义搜索
from llama_index.core.schema import NodeWithScore, TextNode
# 从llama_index库导入RetrieverQueryEngine类，用于协调检索器和响应生成，执行端到端的问答过程
from llama_index.core.query_engine import RetrieverQueryEngine
# 从llama_index库导入QueryBundle类，用于封装查询相关的信息，如查询文本、过滤器等
from llama_index.core import QueryBundle
# 从llama_index库导入BaseRetriever类，这是所有检索器的基类，定义了检索接口
from llama_index.core.retrievers import BaseRetriever
# 从llama_index库导入SentenceSplitter类，用于将长文本分割成句子或语义完整的文本块，便于索引和检索
from llama_index.core.node_parser import SentenceSplitter
# 从llama_index库导入VectorStoreQuery类，用于构造向量存储的查询，支持语义相似度搜索
from llama_index.core.vector_stores import VectorStoreQuery
# 向量数据库
import chromadb
from ipex_llm.llamaindex.llms import IpexLLM

import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import  AutoTokenizer

#自定义
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from typing import List
import shutil

import requests
import certifi

response = requests.get('https://www.modelscope.cn/api/v1/models/Qwen/Qwen2-7B-Instruct/revisions', verify=certifi.where())
print(response.content)



class CustomEmbedding:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_text_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def get_query_embedding(self, query):
        return self.get_text_embedding(query)



class VectorDBRetriever(BaseRetriever):
    """向量数据库检索器"""

    def __init__(
        self,
        vector_store,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        检索相关文档
        
        Args:
            query_bundle (QueryBundle): 查询包
        
        Returns:
            List[NodeWithScore]: 检索到的文档节点及其相关性得分
        """
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        print(f"Retrieved {len(nodes_with_scores)} nodes with scores")
        return nodes_with_scores


class Talk:

    def __init__(self):
        # 第一个参数表示下载模型的型号，第二个参数是下载后存放的缓存地址，第三个表示版本号，默认 master
        snapshot_download('Qwen/Qwen2-7B-Instruct', cache_dir='qwen2chat_src', revision='master')
        model_path = os.path.join(os.getcwd(),"qwen2chat_src/Qwen/Qwen2-7B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='sym_int4', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model.save_low_bit('qwen2chat_int4')
        tokenizer.save_pretrained('qwen2chat_int4')

        # 第一个参数表示下载模型的型号，第二个参数是下载后存放的缓存地址，第三个表示版本号，默认 master
        snapshot_download('AI-ModelScope/bge-small-zh-v1.5', cache_dir='qwen2chat_src', revision='master')


        self.model_path = "qwen2chat_int4"
        self.tokenizer_path = "qwen2chat_int4"
        self.question = "你好"
        self.data_path = "./data/my_data.pdf"
        self.persist_dir = "./chroma_db"
        self.embedding_model_path = "qwen2chat_src/AI-ModelScope/bge-small-zh-v1___5"
        self.max_new_tokens =2048
        self.context_window = 5000

        self.retriever = None

    def load_vector_database(self, persist_dir: str):
        """
        加载或创建向量数据库
        
        Args:
            persist_dir (str): 持久化目录路径
        
        Returns:
            ChromaVectorStore: 向量存储对象
        """
        # 检查持久化目录是否存在
        is_exit = False
        if os.path.exists(persist_dir):
            print(f"正在加载现有的向量数据库: {persist_dir}")
            chroma_client = chromadb.PersistentClient(path=persist_dir)
            chroma_collection = chroma_client.get_collection("llama2_paper")
            is_exit = True
        else:
            print(f"创建新的向量数据库: {persist_dir}")
            chroma_client = chromadb.PersistentClient(path=persist_dir)
            chroma_collection = chroma_client.create_collection("llama2_paper")
        print(f"Vector store loaded with {chroma_collection.count()} documents")
        return chroma_collection, is_exit

    def load_data(self, data_path: str) -> List[TextNode]:
        """
        加载并处理PDF数据
        
        Args:
            data_path (str): PDF文件路径
        
        Returns:
            List[TextNode]: 处理后的文本节点列表
        """
        # 读取TXT文件
        with open(data_path, 'r', encoding='utf-8') as file:
            text = file.read()        
        
        # 初始化 RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 设置合适的 chunk_size
            chunk_overlap=300  # 设置重叠的字符数
        )

        # 分割文本
        text_chunks = text_splitter.split_text(text)

        # 创建 TextNode 对象并添加到列表中
        nodes = [TextNode(text=chunk) for chunk in text_chunks]

        return nodes


    def completion_to_prompt(self, completion: str) -> str:
        """
        将完成转换为提示格式
        
        Args:
            completion (str): 完成的文本
        
        Returns:
            str: 格式化后的提示
        """
        return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"

    def messages_to_prompt(self, messages: List[dict]) -> str:
        """
        将消息列表转换为提示格式
        
        Args:
            messages (List[dict]): 消息列表
        
        Returns:
            str: 格式化后的提示
        """
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"<|system|>\n{message.content}</s>\n"
            elif message.role == "user":
                prompt += f"<|user|>\n{message.content}</s>\n"
            elif message.role == "assistant":
                prompt += f"<|assistant|>\n{message.content}</s>\n"

        if not prompt.startswith("<|system|>\n"):
            prompt = "<|system|>\n</s>\n" + prompt

        prompt = prompt + "<|assistant|>\n"

        return prompt

    def main(self):
        """主函数"""
        
        # 设置嵌入模型
        embed_model = CustomEmbedding(model_name=self.embedding_model_path)
        
                
        # 加载向量数据库
        vector_store, is_exit = self.load_vector_database(persist_dir=self.persist_dir)
        
        # 加载和处理数据
        nodes = self.load_data(data_path=self.data_path)
        for node in nodes:
            node_embedding = embed_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding
        
        # 将 node 添加到向量存储
        if not is_exit:
            vector_store.add(nodes)
        
        # 设置查询
        query_str = self.question
        query_embedding = embed_model.get_query_embedding(query_str)
        
        # 执行向量存储检索
        print("开始执行向量存储检索")
        query_mode = "default"
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
        )
        query_result = vector_store.query(vector_store_query)

        # 处理查询结果
        print("开始处理检索结果")
        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        
        # 设置检索器
        self.retriever = VectorDBRetriever(
            vector_store, embed_model, query_mode="default", similarity_top_k=1
        )
        
        print(f"Query engine created with retriever: {type(self.retriever).__name__}")
        print(f"Query string length: {len(query_str)}")
        print(f"Query string: {query_str}")

    def ask_info(self, query_str, do_sample, temperature):
        # 创建查询引擎
        print("准备与llm对话")

        llm = IpexLLM.from_model_id_low_bit(
            model_name=self.model_path,
            tokenizer_name=self.tokenizer_path,
            context_window=self.config.context_window,
            max_new_tokens=self.config.max_new_tokens,
            generate_kwargs={"temperature": temperature, "do_sample": do_sample},
            model_kwargs={},
            messages_to_prompt=self.messages_to_prompt,
            completion_to_prompt=self.completion_to_prompt,
            device_map="cpu",
        )
        query_engine = RetrieverQueryEngine.from_args(self.retriever, llm=llm)

        # 执行查询
        print("开始RAG最后生成")
        start_time = time.time()
        response = query_engine.query(query_str)

        # 打印结果
        print("------------RESPONSE GENERATION---------------------")
        print(str(response))
        print(f"inference time: {time.time()-start_time}")
        return str(response)

if __name__ == "__main__":
    talk = Talk()
    talk.main()
    # talk.ask_info("How does Llama 2 perform compared to other open-source models?")
    
    # 使用Interface定义页面布局
    demo = gr.Interface(
        fn=talk.ask_info,
        inputs=["text", "checkbox", gr.Slider(0, 1)],
        outputs=["text"],
        allow_flagging='never'
    
    )
    
    # Web UI启动
    demo.launch(inbrowser=True)