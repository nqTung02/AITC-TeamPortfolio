import os
import glob
import shutil
import dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

dotenv.load_dotenv()
CHROMA_DIR = "./chroma_db"
if os.path.exists(CHROMA_DIR):
    print("🗑️ Đang xoá Chroma vectorstore cũ...")
    shutil.rmtree(CHROMA_DIR)

pdf_files = glob.glob("dataset/text/*.pdf")
if not pdf_files:
    raise FileNotFoundError("❌ Không tìm thấy file PDF nào trong dataset/text.")

documents = []

for pdf_path in pdf_files:
    print(f"📄 Đang xử lý: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    full_text = "\n".join([p.page_content for p in pages])
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")])
    split_docs = splitter.split_text(full_text)

    for doc in split_docs:
        doc.metadata["source_file"] = os.path.basename(pdf_path)

    documents.extend(split_docs)

print("📦 Đang tạo vectorstore...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(),
    persist_directory=CHROMA_DIR
)
vectorstore.persist()

print("Hoàn tất xử lý và lưu vectorstore.")
