from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, Document, SimpleNodeParser
import fitz  # PyMuPDF

# 读取和解析PDF文件
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# 读取PDF文件
pdf_path = '/Users/7one/LLM-VL-chatbot/enginedata.pdf'
pdf_text = extract_text_from_pdf(pdf_path)

# 将文本数据转换为节点
documents = [Document(text=pdf_text)]
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# 创建并保存索引
index = GPTVectorStoreIndex(nodes)
index.save_to_disk('engine_diagnostics_index.json')

# 加载索引并进行查询
index = GPTVectorStoreIndex.load_from_disk('engine_diagnostics_index.json')
response = index.query("What are the diagnostic results for the engine?")
print(response)
